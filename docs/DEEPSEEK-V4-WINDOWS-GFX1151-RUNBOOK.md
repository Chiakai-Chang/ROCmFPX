# DeepSeek-V4-Flash-0731 on Windows / Strix Halo (gfx1151)

Measured 2026-08-02 on the reference box: Ryzen AI MAX+ 395, Radeon 8060S
(`gfx1151`), 128 GB unified memory, Windows 11, ROCm 7.1 HIP SDK installed.
The 96/32 carve is **inferred**, not read from the BIOS: Windows reports
32,407 MiB of system RAM, and llama.cpp reports 110,456 MiB for the adapter,
which is the dedicated carve plus WDDM shared memory rather than the carve
itself. Confirm in BIOS before relying on the exact figure.

**Headline: this model does not belong on the ROCmFPX HIP lane. Run it on
Vulkan.** The winning configuration needs no compiler, no ROCm SDK, and nothing
from this repository.

---

## 1. Result

| runner | pp512 | tg128 |
| --- | ---: | ---: |
| **Vulkan, official prebuilt b10221** | **134.89 ± 1.89** | **12.45 ± 0.01** |
| ROCm/HIP, self-built mainline b10223 (gfx1151-only, `FORCE_MMQ=ON`) | 93.48 ± 3.05 | 8.33 ± 0.07 |
| ROCm/HIP, official prebuilt b10227 (10-arch fat binary) | 92.73 ± 0.74 | 8.14 ± 0.13 |
| ROCm/HIP, lemonade `windows-rocm-gfx1151-x64` b10218 | 24.46 ± 0.11 | 7.61 ± 0.04 |
| ROCmFPX fork `c868e96` (this repo) | ~107 @3017 (server) | ~7.5 (server) |
| _Linux Vulkan/RADV reference, same model class_ | _155.64_ | _13.27_ |

`llama-server`, identical flags and prompts, ROCm/HIP -> Vulkan. **These are
single runs.** Repeated `llama-bench` runs on this box vary by about 5%
run-to-run, so read the direction as solid and the magnitude as approximate:

| test | ROCm/HIP | Vulkan | delta |
| --- | ---: | ---: | ---: |
| decode, short context | 8.85 | 12.16 | +37% |
| prefill @3017 tokens | 103.6 | 126.1 | +22% |
| decode @3017 tokens | 6.65 | 11.80 | +77% |
| decode, tool-call turn | 7.86 | 11.78 | +50% |

Windows Vulkan lands close to the published Linux Vulkan/RADV decode figure, so
the original gap was never a Windows limitation and the model never needed
requantising. **Treat that comparison as indicative, not exact**: the Linux
figure is a different artifact (UD-IQ2_XXS, 90.86 GB), a different OS, and a
different build. It is close enough to rule out "Windows is the problem" and not
close enough to quote a percentage from.

## 2. Artifact

`Rednalreden/DeepSeek-V4-Flash-0731-dwarfstar-q2-gguf`, file
`DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf`,
86,720,111,520 bytes / 80.76 GiB, 284.33 B params, 1,328 tensors.

Its model card says "not llama.cpp-compatible". **That is wrong for any build
carrying the `deepseek4` arch** — it means mainline lacked `deepseek4` at
publication. Verified by range-fetching only the GGUF header and reconstructing
every tensor's payload size from offset deltas: all 1,327 adjacent pairs match
standard ggml block layouts byte-for-byte (IQ2_XXS 66 B/256, Q2_K 84 B/256).
No custom packing. That header-range trick costs a few MB of download and is
worth reusing before committing to any large artifact.

Recipe: `IQ2_XXS` routed-expert gate/up, `Q2_K` expert-down, `Q8_0` attention /
shared experts / output head, `F16` hyper-connection, compressor and indexer.
Only the routed experts are 2-bit, which is why the compressor and indexer paths
that carry long context stay at full precision.

## 3. Launchers

Deployed to `C:\models`, one per backend, matching the existing per-model
convention. Each carries its rationale inline with `file:line` citations.

| launcher | backend | use |
| --- | --- | --- |
| `DeepSeek-V4-Flash-0731_vulkan.bat` | Vulkan | **primary** |
| `DeepSeek-V4-Flash-0731_mainline.bat` | ROCm/HIP, mainline | fallback |
| `DeepSeek-V4-Flash-0731_rocm7.bat` | ROCm/HIP, this fork | last resort; only build with `--cache-disk` |

The Vulkan binary lives in `C:\models\llama-bin-win-vulkan-x64` and is kept
current by `C:\models\更新llam.cpp\Update-llama-Vulkan.bat`, which pulls the
official `ggml-org/llama.cpp` `bin-win-vulkan-x64.zip`.

Key flags, all measured rather than assumed:

- `--no-context-shift` — **mandatory**. `llama_kv_cache_dsv4::get_can_shift()`
  returns false whenever the DSV4 compressed cache is live
  (mainline `src/llama-kv-cache-dsv4.cpp:1241`; the fork aborts at
  `src/llama-memory-hybrid-iswa.cpp:399`).
- `-fa off` — **write it explicitly**. Vulkan does have a head_dim 512 FA kernel
  (ROCm does not), but enabling it trades 8.5% of prefill for ~1% of decode:
  `-fa off` 134.89 / 12.45 versus `-fa on` 123.24 / 12.56. `-fa auto` picks FA
  **on**, i.e. the slower path.
- `-b 2048 -ub 512` — measured optimum. Larger ubatch is much worse here
  (pp2048: 111.16 at ub512, 105.71 at ub1024, 82.82 at ub2048). Do not carry
  over the APEX `ub 4096` setting; it is model-specific.
- `--cache-ram 8192` — the highest-value flag for agentic use. A 3,010-token
  prompt resent identically reprocesses 4 tokens (3,006 cached); a follow-up turn
  reprocesses 15. Prefill is paid once per conversation, not once per turn.
  Do **not** add `--cache-reuse`: the server logs `cache_reuse is not supported
  by this context, it will be disabled` for `deepseek4` on both backends, so it
  is inert and only creates the illusion of a tuned setting.
- No `--override-kv` on mainline or Vulkan. The GGUF declares
  `nextn_predict_layers = 1` but contains zero nextn tensors; mainline never
  creates `blk.42.nextn.*` so the metadata is inert. **This fork does** create
  them as required tensors (`src/models/deepseek4.cpp:174-183`) and fails to
  load without `--override-kv deepseek4.nextn_predict_layers=int:0`.
- No `--spec-*`. There are no MTP/nextn tensors, so no speculative decoding path
  exists. Do not copy the Hy3 `draft-mtp` block into these launchers.

## 4. Read this before quoting the headline number

Decode decays with context depth (`llama-bench -d`), and the backend gap widens
exactly where agent sessions live:

| depth | Vulkan | ROCm/HIP |
| ---: | ---: | ---: |
| 0 | 12.40 ± 0.10 | 8.62 ± 0.10 |
| 8,192 | 10.91 (−12%) | 3.76 (−56%) |
| 32,768 | **7.76 (−37%)** | **2.00 (−77%)** |

At 32K Vulkan is 3.9x faster. Plan agentic work at **8–11 t/s**, not 12.4.

**The fused-op situation is the opposite of what the numbers suggest.** The
startup log shows that on Vulkan *all four* fused DeepSeek-V4 ops fall back to
generic kernels:

```
resolve_fused_ops: Lightning Indexer not supported, set to disabled
resolve_fused_ops: fused DeepSeek V4 HC pre  not supported, set to disabled
resolve_fused_ops: fused DeepSeek V4 HC comb not supported, set to disabled
resolve_fused_ops: fused DeepSeek V4 HC post not supported, set to disabled
```

ROCm/HIP emits none of those warnings — it has all four active — and still loses
by the margins above. So the missing Vulkan kernels are **headroom, not a
defect**. The depth cost lives in the unfused `build_lid_top_k` path
(`src/models/deepseek4.cpp:587`), which materialises a full `mul_mat` over every
compressed row plus several `ggml_cont` permutes per layer instead of calling
`ggml_lightning_indexer`. That is a software gap, not a silicon limit; if ggml
gains a Vulkan lightning-indexer kernel, re-measure everything here.

Always read these probe lines at startup. Comparing two backends without them
means comparing two different kernel paths without knowing it.

**Prefill decays far worse, and this is the real usability limit.** A six-turn
session, each turn appending the same ~4.2K-token file, prompt cache on, so the
same number of tokens is reprocessed every turn:

| turn | context | reprocessed | cached | prefill | effective prefill | wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4,219 | 4,219 | 0 | 42.9 s | 98 t/s | 47.4 s |
| 2 | 8,430 | 4,215 | 4,215 | 80.1 s | 53 t/s | 84.6 s |
| 3 | 12,641 | 4,215 | 8,426 | 106.9 s | 39 t/s | 111.8 s |
| 4 | 16,857 | 4,220 | 12,637 | 140.3 s | 30 t/s | 145.4 s |
| 5 | 21,068 | 4,215 | 16,853 | 150.7 s | 28 t/s | 156.0 s |
| 6 | 25,279 | 4,215 | 21,064 | 173.1 s | **24 t/s** | 178.6 s |

Identical work per turn, 4x the time by turn 6. `pp512 134.89` is measured at
depth 0 and is wildly optimistic for real sessions. At ~25K context a 4K-token
turn costs about three minutes. Budget accordingly, and prefer many short
conversations over one long one.

KV costs about 6.7 KiB/token at F16, so 128K context is roughly 0.9 GiB. Weights
plus buffers commit about 85.6 GiB on the adapter, inside the 96 GiB dedicated
carve. Memory is not the constraint; depth-dependent prefill is.

## 4a. Prompt-cache behaviour, measured

`--cache-ram 8192`. Note `--cache-reuse` is **inert** for this model: the server
logs `cache_reuse is not supported by this context, it will be disabled` on both
backends, so everything below comes from whole-prefix `--cache-ram` matching.

- **Pure append works.** Each turn reprocesses only the newly appended tokens.
- **Tool-result append is cheap.** Appending `<tool_result>…</tool_result>` to a
  12.6K-token conversation reprocessed 531 tokens in 7.1 s. The normal agent
  loop is well served.
- **Changing the prefix destroys everything.** Editing the system prompt (adding
  a timestamp) on that same 12.6K conversation reprocessed all 12,674 tokens in
  **225.8 s**. Any harness that injects a clock, a session id, or rotating state
  into the system prompt converts every turn into a full reprocess. On this model
  that is minutes per turn, not seconds. Put volatile state in the *latest* user
  message, never in the system prompt.
- **Multiple prefixes coexist.** Switching back to the original system prompt
  afterwards reprocessed 4 tokens (12,652 cached) in 0.6 s, so the earlier cache
  survived the detour at `--cache-ram 8192`.

## 5. Dead ends, so they are not re-derived

- **Quantized KV on ROCm/HIP is a closed loop.** `-ctk q8_0 -ctv f16` is
  rejected with `model does not support different K (q8_0) and V (f16) cache
  types`; `-ctk q8_0 -ctv q8_0` with `quantized V cache requires flash_attn to
  be enabled`; and ROCm has no gfx1151 FA kernel for head_dim 512.
  **On Vulkan it works**: `-fa on -ctk q8_0 -ctv q8_0` gives pp512 121.09,
  tg128 12.48 — decode unchanged, prefill −10%, KV halved to ~3.6 KiB/token.
  Worth it past roughly 512K context, pointless below.
- **TurboQuant does not help this model.** `GGML_TYPE_TURBO3_0` (105, 3.5 bpw)
  and `GGML_TYPE_TURBO4_0` (106, 4.5 bpw) are fork-only KV-cache types, not
  prefill accelerators, and the fork forces the DSV4 compressed cache to F16
  anyway (`src/llama-memory-hybrid-iswa.cpp:171`). Still useful for Hy3/APEX.
- **"RotoQuant" does not exist** in this fork, in mainline `ggml/` + `src/`, or
  in `llama-cpp-turboquant`.
- **`otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF` (85.26 GiB) cannot be used.** Its
  type 107 is an *affine* `code * scale - offset`; this fork's 107 is
  `Q2_0_ROCMFPX` (symmetric, dual half-block scales) and upstream renumbered 107
  to `Q7_0_ROCMFPX`, moving Q2_0 to 108. It would decode 129 expert tensors to
  garbage **silently**. Its "Ember runtime" is not publicly findable.
- **`mradermacher/DeepSeek-V4-Flash-0731-Abliterated-FP8-GGUF`** is fully
  metadata- and tensor-name-compatible (verified), but its smallest quant is
  Q2_K at 96.13 GiB, which does not fit the dedicated carve.
- **The published "32 tok/s" figures are a different engine.** They come from
  `antirez/ds4` (DwarfStar) on Linux with DSpark speculative decoding and
  `--ds4-expert-top-k 4`; its plain autoregressive baseline is 25.31 t/s. DS4 is
  C99 with a POSIX Makefile and no Windows support, and `ds4fa`'s entire value is
  Linux TTM/GTT tuning that has no Windows equivalent.
- **Do not use the lemonade runner for this model** — pp512 24.46, 3.8x slower
  prefill than either alternative, with error bars far too tight to be noise.

## 6. Lessons

1. **Benchmark every backend before tuning any of them.** Measuring 69% GPU
   utilisation and ~27% of theoretical bandwidth on ROCm, it was tempting to
   conclude the *architecture* was latency-bound — thousands of small kernels
   per token across 43 layers of 20-iteration sinkhorn, compressor and indexer
   top-k, with WDDM launch overhead dominating — and that little more was
   available. That was wrong. Same architecture, same kernels, same WDDM, +49%
   decode on Vulkan. The entire self-built / official-prebuilt / lemonade
   comparison was an intra-ROCm contest over ±5% on the slow path.
2. **Sample depth, not just peak.** Measuring only to 3,017 tokens produced
   "Vulkan barely decays with context". At 32K it is −37%.
3. **`llama-bench.exe` outlives its parent task.** A stale process held 82.52 GiB
   of GPU memory after its background task reported completion, and the next run
   failed with `ggml_vulkan: Device memory allocation of size 704643072 failed /
   ErrorOutOfDeviceMemory` — which reads like "unsupported config", not "the GPU
   is occupied". Check `Get-Process llama-bench,llama-server` and
   `Get-Counter "\GPU Adapter Memory(*)\Total Committed"` (a clean desktop sits
   near 2.3 GiB here) before believing any OOM, and re-run the baseline after
   cleanup rather than assuming earlier numbers were unaffected.

## 7. Verdict on fitness for use

It runs, it answers coherently, it emits well-formed DSML tool calls, and it fits
the carve with room to spare. But depth-dependent prefill makes long agentic
sessions expensive: roughly three minutes per 4K-token turn once the
conversation reaches ~25K. It is well suited to short, self-contained tasks and
to batch work where latency does not matter, and poorly suited to interactive
long-context agent loops. Whether that trade is worth it against the smaller
models already in the roster is a judgement call that needs a quality comparison
that has not been run yet — see below.

## 8. Still open

- Quality of the 2-bit routed experts has not been evaluated at all. Everything
  above establishes that it runs and answers coherently, not that it is good.
  The publisher reports a 1.42% relative NLL improvement from their imatrix
  against a no-imatrix build of the same checkpoint, which says nothing about
  absolute quality.
- No speculative decoding is possible: no MTP tensors in this GGUF, mainline
  does not implement DSpark, and no same-vocabulary draft model is available.
- The residual ~6.5% decode gap to Linux Vulkan/RADV is unattributed.
- Mainline dropped `--cache-disk` / `--cache-disk-limit` (SSD-backed prompt
  cache); only `--cache-ram` remains. This fork still has it. Not yet assessed
  whether RAM-only caching is sufficient for the intended agent workload.
