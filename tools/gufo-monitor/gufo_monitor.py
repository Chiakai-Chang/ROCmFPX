#!/usr/bin/env python3
"""Read-only terminal monitor for a Gufo server (Windows or Linux, needs only `rich`).

Sources, none of which touch the server's state:
  * HTTP  /health, /slots, /metrics  (polled once a second; shows "offline" when the server is down)
  * the newest server log (progress lines, cache events, completed requests, device_lost)
  * host memory from the OS

Usage:  python -X utf8 gufo_monitor.py [--url http://127.0.0.1:8080] [--logs D:\\Gufo-Logs] [--ascii]
        python -X utf8 gufo_monitor.py --once      # print one frame and exit (also what the self-test uses)
Keys:   q quit   space freeze   r reset history
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import glob
import json
import os
import re
import sys
import time
import urllib.request

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROGRESS = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[\w+\] \[progress\] request=(?P<req>\d+) "
    r"phase=(?P<phase>prefill|decode) tokens=(?P<tok>\d+)/(?P<total>\d+) percentage=(?P<pct>[\d.]+) "
    r"chunk_tps=(?P<chunk>[\d.]+) avg_tps=(?P<avg>[\d.]+)"
    r"(?: draft_accepted=(?P<acc>\d+) draft_proposed=(?P<prop>\d+) acceptance_percentage=(?P<accpct>[\d.]+))?")
COMPLETED = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .*request=(?P<id>r\d+) event=completed method=(?P<m>\w+) "
    r"path=(?P<path>\S+) status=(?P<status>\d+) duration_ms=(?P<ms>[\d.]+) outcome=(?P<outcome>\w+)"
    r"(?:.*rss_mib=(?P<rss>\d+))?(?:.*host_available_mib=(?P<avail>\d+))?")
ACQUIRE = re.compile(r"event=acquire reuse=(?P<reuse>\w+) selected_tokens=(?P<sel>\d+)")
PUBLISHED = re.compile(r"event=published tokens=(?P<tok>\d+)")
WATCH = ("capture_refused", "persistence_incomplete", "device_lost", "snapshot_skipped")
CHAT_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages", "/completion", "/v1/completions")


class State:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.decode = collections.deque(maxlen=60)
        self.prefill = collections.deque(maxlen=60)
        self.last_progress: dict | None = None
        self.last_progress_at = 0.0
        self.last_decode_avg = 0.0
        self.last_prefill_avg = 0.0
        self.accept_pct: float | None = None
        self.cache_last = ""
        self.published = 0
        self.watch = collections.Counter()
        self.requests: collections.deque = collections.deque(maxlen=8)
        self.rss = self.avail_log = None
        self.log_name = ""

    def feed(self, line: str, stamp: float) -> None:
        m = PROGRESS.match(line)
        if m:
            d = m.groupdict()
            tps = float(d["chunk"])
            (self.decode if d["phase"] == "decode" else self.prefill).append(tps)
            self.last_progress = d
            if d["phase"] == "decode":
                self.last_decode_avg = float(d["avg"])
            if d["accpct"] is not None:
                self.accept_pct = float(d["accpct"])
            self.last_progress_at = stamp
            if d["phase"] == "prefill":
                self.last_prefill_avg = float(d["avg"])
            return
        m = COMPLETED.match(line)
        if m:
            d = m.groupdict()
            if d["rss"]:
                self.rss = int(d["rss"])
            if d["avail"]:
                self.avail_log = int(d["avail"])
            if d["m"] == "POST" and d["path"].startswith(CHAT_PATHS):
                lp = self.last_progress or {}
                self.requests.appendleft({
                    "time": d["ts"][11:], "status": d["status"], "outcome": d["outcome"],
                    "dur": float(d["ms"]) / 1000.0, "tok": lp.get("tok", ""), "avg": lp.get("avg", ""),
                    "cache": self.cache_last})
            return
        m = ACQUIRE.search(line)
        if m:
            self.cache_last = f"{m['reuse']} {int(m['sel']):,}"
            return
        m = PUBLISHED.search(line)
        if m:
            self.published += 1
            return
        for key in WATCH:
            if f"event={key}" in line:
                self.watch[key] += 1


class LogTail:
    """Follows the newest log in a directory; seeds from the last `seed` bytes."""

    def __init__(self, directory: str, state: State, seed: int = 400_000) -> None:
        self.dir, self.state, self.seed = directory, state, seed
        self.path = ""
        self.fh = None

    def newest(self) -> str:
        files = glob.glob(os.path.join(self.dir, "*.log"))
        return max(files, key=os.path.getmtime) if files else ""

    def poll(self) -> None:
        path = self.newest()
        if not path:
            return
        mtime = os.path.getmtime(path)
        if path != self.path:
            if self.fh:
                self.fh.close()
            self.path, self.state.log_name = path, os.path.basename(path)
            self.fh = open(path, "r", encoding="utf-8", errors="replace")
            size = os.path.getsize(path)
            if size > self.seed:
                self.fh.seek(size - self.seed)
                self.fh.readline()
            seeding = True
        else:
            seeding = False
        for line in self.fh:
            self.state.feed(line.rstrip("\n"), stamp=mtime if seeding else time.time())


def http_get(url: str, timeout: float = 1.5) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - offline is a normal state
        return None


def parse_metrics(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.partition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def host_memory() -> tuple[float, float] | None:
    """(available GiB, total GiB)."""
    if os.name == "nt":
        class MEM(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
        m = MEM()
        m.l = ctypes.sizeof(MEM)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.avail / 2**30, m.total / 2**30
        return None
    try:
        info = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        return info["MemAvailable"] / 2**20, info["MemTotal"] / 2**20
    except Exception:  # noqa: BLE001
        return None


def spark(values, width: int, ascii_only: bool) -> str:
    chars = " .:-=+*#%@" if ascii_only else " ▁▂▃▄▅▆▇█"
    vals = list(values)[-width:]
    if not vals:
        return ""
    hi = max(max(vals), 1e-9)
    return "".join(chars[min(len(chars) - 1, int(v / hi * (len(chars) - 1) + 0.5))] for v in vals)


def bar(frac: float, width: int, ascii_only: bool) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width + 0.5)
    full, empty = ("#", ".") if ascii_only else ("█", "░")
    return full * n + empty * (width - n)


def card(title: str, big: str, sub: str, trend: str, style: str) -> Panel:
    body = Text(no_wrap=True, overflow="ellipsis")
    body.append(big + "\n", style=f"bold {style}")
    body.append(sub + "\n", style="dim")
    body.append(trend, style=style)
    return Panel(body, title=title, border_style="grey50")


def render(state: State, url: str, ascii_only: bool, frozen: bool, snap: dict, width: int = 118) -> Group:
    inner = max(8, (width - 4) // 4 - 4)     # four cards a row; borders and padding take 4 columns each
    full = max(10, min(70, width - 2))
    now = time.time()
    online = snap["health"] is not None
    p = state.last_progress
    active = bool(p) and now - state.last_progress_at < 6 or snap.get("processing", False)
    if not online:
        mode, style = "OFFLINE", "red"
    elif active and p and p["phase"] == "prefill":
        mode, style = "PREFILL", "yellow"
    elif active:
        mode, style = "DECODE", "green"
    else:
        mode, style = "IDLE", "cyan"
    head = Text()
    head.append(" GUFO ", style="bold black on white")
    head.append(f"  {url}  ", style="bold")
    head.append(f"{mode}", style=f"bold {style}")
    if snap.get("model"):
        head.append(f"  {snap['model']}", style="dim")
    tag, tag_style = ("[FROZEN]", "bold yellow") if frozen else ("[STALE: log only]", "bold red") if not online else ("[LIVE]", "bold green")
    head.append("   " + tag, style=tag_style)

    states = Text("State  ")
    for name in ("IDLE", "PREFILL", "DECODE", "OFFLINE"):
        states.append(f"[{name}] " if name == mode else f" {name}  ", style=f"bold {style}" if name == mode else "dim")
    fresh = now - state.last_progress_at < 6
    if active and p and fresh:
        states.append(f"   {int(p['tok']):,} / {int(p['total']):,} tokens  {float(p['pct']):.0f}%", style="bold")
        prog = bar(float(p["pct"]) / 100, full, ascii_only)
    else:
        prog = bar(0, full, ascii_only)

    dec_now = state.decode[-1] if state.decode else 0.0
    pre_now = state.prefill[-1] if state.prefill else 0.0
    acc = state.accept_pct
    m = snap.get("metrics", {})
    drafts = m.get("llamacpp:spec_decode_num_draft_tokens_total", 0)
    accepted = m.get("llamacpp:spec_decode_num_accepted_tokens_total", 0)
    life_acc = f"lifetime {accepted / drafts * 100:.0f}%" if drafts else "lifetime n/a"
    mem = host_memory()
    avail = mem[0] if mem else (state.avail_log / 1024 if state.avail_log else None)
    ctx_used, ctx_total = snap.get("ctx_used", 0), snap.get("n_ctx", 0)
    lost = int(m.get("gufo_device_lost_total", 0)) or state.watch.get("device_lost", 0)

    row1 = [
        card("Decode", f"{dec_now:.1f} tok/s" if state.decode else "-", f"avg {state.last_decode_avg:.1f} /run",
             spark(state.decode, inner, ascii_only), "green"),
        card("Prefill", f"{pre_now:.0f} tok/s" if state.prefill else "-", f"avg {state.last_prefill_avg:.0f} /prompt",
             spark(state.prefill, inner, ascii_only), "yellow"),
        card("MTP accept", f"{acc:.0f} %" if acc is not None else "-", life_acc,
             bar((acc or 0) / 100, inner, ascii_only), "cyan"),
        card("Device lost", str(lost), "since start", "", "red" if lost else "green"),
    ]
    rss_gib = (state.rss or 0) / 1024
    row2 = [
        card("Host RAM free", f"{avail:.1f} GiB" if avail is not None else "-",
             f"of {mem[1]:.0f} GiB" if mem else "from log",
             bar(1 - (avail / mem[1]) if avail is not None and mem else 0, inner, ascii_only),
             "red" if avail is not None and avail < 4 else "green"),
        card("Server RSS", f"{rss_gib:.1f} GiB" if state.rss else "-", "last request", "", "magenta"),
        card("Context", f"{ctx_used:,}" if ctx_total else "-", f"of {ctx_total:,} live" if ctx_total else "no /slots",
             bar(ctx_used / ctx_total if ctx_total else 0, inner, ascii_only), "blue"),
        card("Cache", state.cache_last or "-", f"published {state.published}", "", "white"),
    ]

    def grid(cards):
        t = Table.grid(expand=True)
        for _ in cards:
            t.add_column(ratio=1)
        t.add_row(*cards)
        return t

    watch = "  ".join(f"{k}={v}" for k, v in state.watch.items() if k != "device_lost") or "none"
    table = Table(title="Recent requests", title_justify="left", box=None, header_style="bold")
    for col in ("TIME", "STATUS", "OUTCOME", "TOKENS", "AVG TPS", "DURATION", "CACHE"):
        table.add_column(col, justify="right" if col in ("TOKENS", "AVG TPS", "DURATION") else "left")
    for r in state.requests:
        ok = r["status"] == "200"
        table.add_row(r["time"], Text(r["status"], style="green" if ok else "red"), r["outcome"],
                      f"{int(r['tok']):,}" if str(r["tok"]).isdigit() else "-",
                      f"{float(r['avg']):.1f}" if r["avg"] else "-", f"{r['dur']:.1f} s", r["cache"] or "-")
    foot = Text(f" log {state.log_name or '-'}   events: {watch}", style="dim", no_wrap=True, overflow="ellipsis")
    keys_line = Text(" q quit   space freeze   r reset   (read only)", style="dim", no_wrap=True)
    return Group(head, Text(""), states, Text(prog, style=style), Text(""), grid(row1), grid(row2), Text(""), table, foot, keys_line)


def snapshot(base: str) -> dict:
    snap: dict = {"health": http_get(base + "/health"), "metrics": {}}
    if snap["health"] is None:
        return snap
    txt = http_get(base + "/metrics")
    if txt:
        snap["metrics"] = parse_metrics(txt)
    raw = http_get(base + "/slots")
    if raw:
        try:
            slots = json.loads(raw)
            snap["n_ctx"] = max((s.get("n_ctx", 0) for s in slots), default=0)
            snap["ctx_used"] = max((s.get("n_prompt_tokens", 0) + s.get("next_token", [{}])[0].get("n_decoded", 0)
                                    for s in slots), default=0)
            snap["processing"] = any(s.get("is_processing") for s in slots)
            snap["model"] = slots[0].get("model", "") if slots else ""
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    return snap


def keys() -> str:
    if os.name != "nt":
        return ""
    import msvcrt
    out = ""
    while msvcrt.kbhit():
        out += msvcrt.getwch()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("GUFO_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--logs", default=os.environ.get("GUFO_LOGS", r"D:\Gufo-Logs"))
    ap.add_argument("--ascii", action="store_true", help="ASCII glyphs only (old consoles)")
    ap.add_argument("--once", action="store_true", help="render one frame and exit")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    base = a.url.rstrip("/")
    state, tail = State(), None
    tail = LogTail(a.logs, state)
    console = Console()
    tail.poll()
    if a.once:
        console.print(render(state, base, a.ascii, False, snapshot(base), console.width))
        return 0
    frozen, snap, last_snap = False, {"health": None, "metrics": {}}, 0.0
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            k = keys().lower()
            if "q" in k:
                return 0
            if " " in k:
                frozen = not frozen
            if "r" in k:
                state.reset()
            if not frozen:
                tail.poll()
                if time.time() - last_snap > 1.0:
                    snap, last_snap = snapshot(base), time.time()
            live.update(render(state, base, a.ascii, frozen, snap, console.width))
            time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
