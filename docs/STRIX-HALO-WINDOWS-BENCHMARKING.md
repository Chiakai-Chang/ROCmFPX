# Benchmarking on Windows / Strix Halo (gfx1151)

How to get numbers off this hardware that survive being quoted, and what the
numbers said the last time they were taken. Companion to
[DEEPSEEK-V4-WINDOWS-GFX1151-RUNBOOK.md](DEEPSEEK-V4-WINDOWS-GFX1151-RUNBOOK.md),
which covers one model in depth; this covers the method and the backend choice.

Reference box: ASUS ProArt PX13 HN7306EAC, Ryzen AI MAX+ 395, Radeon 8060S
(`gfx1151`), 128 GB unified memory carved 96 GB to the GPU, Windows 11, ROCm 7.1
HIP SDK. **IOMMU is enabled and the BIOS exposes no option to change it** — see
§5 before spending time on that lever.

---

## 1. The measurement rules

These are ordered by how much damage they prevent.

**Run-to-run noise is about 7%, and it lives between `llama-bench` processes,
not inside them.** The same configuration measured twice:

| run | config | pp512 @ d0 |
| --- | --- | ---: |
| 1 | `-ub 2048 -fa on` | 977.04 ± 17.07 |
| 2 | `-ub 2048 -fa on` | 914.62 ± 26.78 |

The reported `±` is the spread across `-r` repetitions inside one process, and
it badly understates real uncertainty: those two intervals do not overlap.
**Only compare arms measured inside a single `llama-bench` invocation.** Any
cross-invocation difference under ~7% on a shallow prompt is nothing. This is
also why a tuning result should be reproduced before it is written into a
launcher.

**Sweep depth. A single shallow point is not a curve.** `-d 0,8192,32768` is
the minimum. DeepSeek-V4 looked like it "barely decays with context" when
measured only to 3K; at 32K it is −37%. Backends do not merely differ in level,
they differ in slope — see §2, where the gap between two ROCm builds reverses
between d0 and d32768.

**The first depth measured absorbs cold start.** Vulkan shader JIT plus the
first weight and KV allocation land on whichever test runs first. A table where
4K prefill is *slower* than 8K is showing that artifact, not the hardware. Warm
up, or discard the first point.

**Write `-fa on` explicitly; never leave it at `auto`.** On DeepSeek-V4 `auto`
selected the slower path. On the qwen35moe model in §2, turning FA off costs
−16% prefill and −15% decode at depth:

| ubatch | pp512 @ d32768 `-fa on` | `-fa off` | tg128 @ d32768 `-fa on` | `-fa off` |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 533.17 ± 6.60 | 447.71 ± 2.97 | 52.29 ± 0.13 | 44.50 ± 0.06 |
| 1024 | 534.39 ± 4.91 | 441.21 ± 0.45 | 52.04 ± 0.08 | 44.14 ± 0.13 |
| 2048 | 527.51 ± 7.16 | 444.06 ± 0.79 | 51.76 ± 0.13 | 44.05 ± 0.36 |

**`-ub` has no portable value.** Measured optima on this box range from 512 to
4096 depending on the model. For the model in §2 it is 512 (pp512 @ d0: 971.46
at ub512 vs 916.42 at ub1024 vs 914.62 at ub2048; all three tie at d32768 and
decode is flat). Do not carry a ubatch across models.

**Check the GPU is idle before believing any number — including any OOM.** A
`llama-bench` process outlives the shell that started it. One was found holding
82.52 GiB after its task had reported completion, and the next run failed with
`ggml_vulkan: Device memory allocation of size 704643072 failed /
ErrorOutOfDeviceMemory`, which reads like "this configuration is unsupported"
rather than "the GPU is occupied".

```powershell
Get-Process llama-server,llama-bench -ErrorAction SilentlyContinue
(Get-Counter "\GPU Adapter Memory(*)\Total Committed").CounterSamples |
    Where-Object CookedValue -gt 500MB
```

A clean desktop sits near 3.0 GiB on this box. Kill leftovers, re-check, and
re-run the *baseline* rather than assuming earlier numbers were unaffected.

**`llama-bench` does not prove a launcher works.** It never allocates the
server's compute buffers. Acceptance is: start `llama-server`, wait for
`/v1/chat/completions` to return 200 — `/props` answers while the model is
still loading and the chat endpoint 503s at that point — then compare GPU
committed memory against your own weight + KV arithmetic.

**Read the startup log every time.** A flag being accepted is not a flag being
active, and the interesting failures are silent degradations rather than errors:

- `--reasoning-preserve` is a no-op on templates that do not support it, and the
  server says so; on a template that does, it volunteers
  `chat template supports preserving reasoning`.
- `--mmproj` silently disables `--cache-reuse`.
- `--no-mmap` is deprecated in current builds in favour of `--load-mode none`
  (not memory-mapping still matters on this UMA box — the mmap loader stalls).
- `resolve_fused_ops:` lines tell you whether two backends are even running the
  same kernel path.

---

## 2. Backend selection is per model, and the old rules expire

Measured 2026-08-03. `GRM-3.2-Sky` OPAL-balanced, a 35B-A3B `qwen35moe` hybrid
SSM MoE (256 experts, 8 used, 23.75 GiB), flags `-ngl 999 -fa on -b 4096
-ub 2048 -lm none -r 3`, GPU verified idle, one runner at a time.

| test | official Vulkan | lemonade ROCm gfx1151 | official HIP |
| --- | ---: | ---: | ---: |
| pp512 | **977.04 ± 17.07** | 836.97 ± 36.72 | 825.36 ± 39.40 |
| pp512 @ d8192 | **798.77 ± 15.79** | 665.03 ± 8.75 | 682.66 ± 8.56 |
| pp512 @ d32768 | **552.03 ± 3.52** | 450.13 ± 19.62 | 508.64 ± 15.04 |
| tg128 | **64.15 ± 0.87** | 57.00 ± 0.68 | 56.01 ± 0.42 |
| tg128 @ d8192 | **58.99 ± 1.06** | 54.40 ± 0.22 | 53.74 ± 0.39 |
| tg128 @ d32768 | **52.34 ± 0.52** | 48.46 ± 0.17 | 47.89 ± 0.13 |

Vulkan wins both axes at every depth, by 16.7 / 20.1 / 22.6% on prefill over the
gfx1151-specific ROCm build.

Two things worth extracting:

**A backend rule derived from one architecture does not transfer.** The prior
rule on this box was "dense models to the gfx1151 ROCm build, MoE to Vulkan",
derived from a dense 27B. Here the gfx1151 build loses to the *generic* official
HIP build at depth (450 vs 509 @ d32768) while winning at d0 — the two ROCm
builds trade places as depth grows. A single-depth comparison would have
reported the opposite conclusion.

**State the confound rather than hiding it.** The Vulkan and official HIP
binaries are both `0b14b87d7` (b10240); the lemonade build is `c745be2` on its
own numbering and predates them. How much of its loss is tuning versus engine
age is *not* separable from this data. The measurement is still decisive about
which runner to use today; it is not evidence about which tuning approach is
better.

---

## 3. Verify the GGUF before benchmarking it

Two defects in this class were hit within a week, both cheap to detect and
expensive to debug from the other end.

**Inconsistent metadata makes a model unloadable on every build.** Four
independently published GGUFs of the same model declared `block_count 41` and
`nextn_predict_layers 1` while shipping 40 blocks, so every runner died with
`missing tensor 'blk.40.attn_norm.weight'`. This is not fixable with flags, and
"try a different quant" does not help when the whole conversion pipeline shares
the defect. A header read finds it in seconds.

**Quant tier names are not a contract.** In one repository the tier named
`quality` was *lower* precision than the one named `balanced` — the two files
were tensor-for-tensor identical except that the middle-layer experts were
`IQ4_XS` in `quality` and `Q5_K` in `balanced`. Compare the per-layer tensor
types, not the labels, and not `general.file_type` (which is a single nominal
label over a mixed recipe).

What to check before writing a launcher:

- `block_count` against the actual `blk.N.*` range, and that the range has no
  gaps
- `nextn_predict_layers` against the presence of real `nextn` tensors — the
  metadata alone decides nothing, and quantizers do strip MTP layers
- per-layer precision of the expert tensors
- the file's SHA256 against whatever the publisher states

`gguf-py`'s `gguf_dump.py` does this, but imports `numpy`. Reading the header
directly is ~80 lines of pure Python (magic, version, tensor count, KV count,
then KV pairs and tensor infos) and is worth keeping around for machines without
a scientific Python stack. For a remote file, the DeepSeek-V4 runbook documents
verifying a multi-GB artifact by range-fetching only its header.

---

## 4. Sizing: attention interval dominates KV, not layer count

Hybrid SSM models pay KV on only a fraction of their layers. For the model in
§2, `full_attention_interval 4` means 10 of 40 layers are real attention:

```
10 layers x 2 kv heads x (k 256 + v 256) x 2 B (f16) = 20,480 B/token = 20 KiB
20 KiB x 262,144 = 5.00 GiB
```

Weights 23.75 GiB + KV 5.00 GiB ≈ 29 GiB at the full native 262K context.
Verified against reality: GPU Adapter Total Committed read 32.43 GiB against a
3.04 GiB desktop baseline. **Quantizing KV on such a model saves nothing** —
compute the number before trading precision for it. A conventional attention
model of similar size on this box pays 64 KiB/token, i.e. 16 GiB at 262K, where
the trade is real.

---

## 5. Levers already investigated, so they are not re-derived

**IOMMU — real, but not available on every board.** Third-party A/B on a
Minisforum MS-S1 MAX reports that disabling IOMMU is worth 10–20% prefill at 32K
and 30–40% at 128K, decode unaffected (the cost is HIP DMA translation, which
only touches compute-bound prefill). The ASUS ProArt PX13 BIOS exposes no IOMMU
or AMD CBS entry, so this is unavailable on the reference box. To determine the
state on any Windows machine, enumerate ACPI tables via
`EnumSystemFirmwareTables('ACPI')` and look for **IVRS** — firmware publishes it
only when AMD-Vi is on. Corroborating signals: `Win32_DeviceGuard`
`AvailableSecurityProperties` containing 3 (DMA protection), and System Guard
Secure Launch running. Note that disabling it also drops Kernel DMA Protection.

**`GGML_HIP_ROCWMMA_FATTN` — do not enable.** Previously listed here as an
untested prefill lever. The justinappler `llama.cpp-strix-halo` fork measured it
as neutral on landing and later actively harmful (−34% prefill at 16K) once
upstream routed head sizes > 128 to the TILE kernel, and the Windows HIP SDK
7.1.1 ships no rocWMMA headers in any case.

**MMQ tile/nwarp tuning for gfx1151 — the live candidate.** The same fork
reports that the generic RDNA3 tile sizes are too large for this part's 40 CUs
and 8 MB L2, and that retuning them (upstream PR #21344) is worth +27% prefill
at d0 and +17% at 16K, with a further +6.3% at 16K from routing FA D=256 to the
TILE kernel. Their published comparison claims prefill 7–43% above Vulkan on a
35B-A3B model class, which is exactly the class measured in §2 — so §2's Vulkan
figures are a usable baseline for evaluating that claim on this box.

**ROCm version landscape.** gfx1151 has a VGPR-count bug in ROCm 6.x–7.1.x
(ROCm/TheRock#2991) fixed in 7.2, while the Windows HIP SDK is pinned at 7.1.1;
some official packages carry broken gfx1151 kernel artifacts (ROCm/ROCm#6042)
that surface as HSA `Invalid Kernel Image`. Third-party numbers headlining
TheRock nightlies are Linux numbers and are not reachable from the Windows SDK.

Claims in this section that are not marked as measured on the reference box are
third-party reports, recorded so they can be tested rather than assumed.

---

## 6. Building on Windows

See [../win-hip-stl-shim/BUILD-WINDOWS.md](../win-hip-stl-shim/BUILD-WINDOWS.md).
One point is relevant to anyone building a gfx1151-tuned tree, not just this
fork: the MSVC STL / ROCm clang `isgreater` collision has two known fixes, and
the shim in that directory is the better one. Suppressing it with
`-fno-cuda-host-device-constexpr` instead breaks `initializer_list` range-for in
device code, which then has to be worked around by editing the fattn kernels —
independently confirmed by the justinappler write-up, which took that route and
had to patch two sites in `ggml/src/ggml-cuda/fattn-mma-f16.cuh`.
