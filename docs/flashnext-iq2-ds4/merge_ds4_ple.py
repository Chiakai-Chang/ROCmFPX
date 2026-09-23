#!/usr/bin/env python3
"""Merge a DS4 "light" Qwen3.8-Flash-Next main GGUF with its PLE sidecar into one llama.cpp GGUF.

DS4 packs (antirez/ds4 format) store the n-gram PLE table in a separate sidecar file and use a
few metadata encodings this fork's qwen4exp loader does not accept. This script:

  - copies the main file's KV section, rewriting `attention.compress_ratios` and `ple.layers`
    from u64 arrays to u32 arrays (the loader reads them as 32-bit)
  - adds `qwen4exp.rope.dimension_sections` = [11, 11, 10, 0] (mrope_section from the
    Qwen/Qwen3.8-Flash-Next config.json; the DS4 pack omits it)
  - appends the sidecar tensor `ple.weight` as `per_layer_token_embd.weight`
  - copies all tensor data byte for byte (nothing is re-quantized)

The padded Q2_K down experts ([768, 2560, 512], input padded 640 -> 768) are kept as they are;
loading them needs commit 14f2af2c5 of this branch.

usage: merge_ds4_ple.py MAIN.gguf PLE_SIDECAR.gguf OUT.gguf [--header-only]
--header-only writes only the new header (for checking); the result is not a loadable model.
"""
import argparse
import os
import struct

# element byte sizes for GGUF scalar value types
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_ARRAY, GGUF_STRING, GGUF_U32, GGUF_I32, GGUF_U64 = 9, 8, 4, 5, 10
GGML_TYPE_Q4_1 = 3

TO_U32 = {"qwen4exp.attention.compress_ratios", "qwen4exp.ple.layers"}
EXTRA_KV = [
    ("qwen4exp.rope.dimension_sections",
     struct.pack("<IIQ", GGUF_ARRAY, GGUF_I32, 4) + struct.pack("<4i", 11, 11, 10, 0)),
]


class Gguf:
    """Minimal GGUF v3 header reader that records byte offsets."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        f = self.f
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        self.version, = struct.unpack("<I", f.read(4))
        if self.version != 3:
            raise ValueError(f"{path}: GGUF version {self.version}, expected 3")
        self.n_tensors, self.n_kv = struct.unpack("<QQ", f.read(16))
        self.align = 32
        self.kvs = []  # (key, type, start, value_start, end)
        for _ in range(self.n_kv):
            p0 = f.tell()
            key = self._str()
            t, = struct.unpack("<I", f.read(4))
            p1 = f.tell()
            v = self._skip_value(t)
            self.kvs.append((key, t, p0, p1, f.tell()))
            if key == "general.alignment":
                self.align = v
        self.tinfo_start = f.tell()
        self.tensors = []  # (name, dims, type, offset)
        for _ in range(self.n_tensors):
            name = self._str()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            typ, off = struct.unpack("<IQ", f.read(12))
            self.tensors.append((name, dims, typ, off))
        self.tinfo_end = f.tell()
        self.data_start = (self.tinfo_end + self.align - 1) // self.align * self.align
        self.size = os.path.getsize(path)

    def _str(self):
        n, = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8")

    def _skip_value(self, t):
        f = self.f
        if t == GGUF_STRING:
            return self._str()
        if t == GGUF_ARRAY:
            et, n = struct.unpack("<IQ", f.read(12))
            if et == GGUF_STRING:
                for _ in range(n):
                    self._str()
            else:
                f.seek(SZ[et] * n, 1)
            return None
        raw = f.read(SZ[t])
        return int.from_bytes(raw, "little") if t in (0, 2, 4, 10) else raw


def copy_bytes(src, dst, n, chunk=64 << 20):
    left = n
    while left:
        b = src.read(min(chunk, left))
        if not b:
            raise EOFError("unexpected end of input")
        dst.write(b)
        left -= len(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("main")
    ap.add_argument("sidecar")
    ap.add_argument("out")
    ap.add_argument("--header-only", action="store_true")
    args = ap.parse_args()

    m, sc = Gguf(args.main), Gguf(args.sidecar)
    if any(t[0] == "per_layer_token_embd.weight" for t in m.tensors):
        raise SystemExit("main file already contains per_layer_token_embd.weight")
    ple = [t for t in sc.tensors if t[0] == "ple.weight"]
    if len(ple) != 1:
        raise SystemExit("sidecar has no ple.weight tensor")
    _, dims, typ, soff = ple[0]
    if typ != GGML_TYPE_Q4_1 or dims[0] % 32:
        raise SystemExit(f"unexpected ple.weight type {typ} / dims {dims}; only Q4_1 is handled")
    ple_bytes = dims[0] // 32 * 20 * dims[1]  # Q4_1: 32 values -> 20 bytes

    main_data = m.size - m.data_start
    new_off = (main_data + m.align - 1) // m.align * m.align
    name = b"per_layer_token_embd.weight"
    tinfo = (struct.pack("<Q", len(name)) + name + struct.pack("<I", len(dims)) +
             struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, new_off))

    m.f.seek(0)
    hdr = m.f.read(m.tinfo_end)
    with open(args.out, "wb") as out:
        out.write(b"GGUF" + struct.pack("<IQQ", 3, m.n_tensors + 1, m.n_kv + len(EXTRA_KV)))
        for key, t, p0, p1, p2 in m.kvs:
            if key in TO_U32:
                et, n = struct.unpack_from("<IQ", hdr, p1)
                if t != GGUF_ARRAY or et != GGUF_U64:
                    raise SystemExit(f"{key}: expected a u64 array")
                vals = struct.unpack_from(f"<{n}Q", hdr, p1 + 12)
                out.write(hdr[p0:p1] + struct.pack("<IQ", GGUF_U32, n) + struct.pack(f"<{n}I", *vals))
            else:
                out.write(hdr[p0:p2])
        for key, value in EXTRA_KV:
            kb = key.encode()
            out.write(struct.pack("<Q", len(kb)) + kb + value)
        out.write(hdr[m.tinfo_start:m.tinfo_end])
        out.write(tinfo)
        if args.header_only:
            return
        out.write(b"\0" * ((-out.tell()) % m.align))
        m.f.seek(m.data_start)
        copy_bytes(m.f, out, main_data)
        out.write(b"\0" * (new_off - main_data))
        sc.f.seek(sc.data_start + soff)
        copy_bytes(sc.f, out, ple_bytes)
    print(f"wrote {args.out}: {m.n_tensors + 1} tensors, {os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
