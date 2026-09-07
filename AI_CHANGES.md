# ROCmFPX — AI Change Log & Handoff

> **Instructions for any AI assistant reading this repository:**
>
> This file is the authoritative record of AI-assisted changes made to ROCmFPX.
> **Every time you make a meaningful change to any file in this repository, you must add an entry to the bottom of this file** under the next numbered session heading. Include: date, files touched, what changed, and why. Keep entries factual and terse — this is a technical log, not a narrative.
>
> Do not rewrite or remove existing entries. Append only.

---

## Session 001 — 2026-06-21

**Scope:** Audit and optimisation pass across `eagle3.cpp`, `rocmfpx.c`, and `scripts/rocmfpx-draft-profile.py`.

### `src/models/eagle3.cpp`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Log typo | 28 | `"EAGLE3gnorm_before_residual"` → `"EAGLE3 norm_before_residual"` (missing space broke log grep) |
| Style | 113 | Added missing space in `ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,hparams…)` call |
| Concat dimension | 217 | Replaced loop variable `il` with explicit constant `0` as the `ggml_concat` dimension argument. `il` was always 0 for the single-layer EAGLE3, but using a bare loop variable made the intent invisible and would silently break if `n_layer > 1` were ever added. |
| Rename | 214, 257 | `inpSA` → `residual`. The variable is the residual connection target, not the self-attention input. Added clarifying comment explaining `norm_before_residual` behaviour. |

---

### `scripts/rocmfpx-draft-profile.py`

| Fix | Detail |
|-----|--------|
| Tokenize timeout | `urllib.urlopen(..., timeout=300)` → `timeout=10`. A stalled server previously hung the client for 5 minutes. |
| `p_split` emitted | All returned profile dicts now include `"speculative.p_split"`. Previously the key was absent, so `dense-coder` at `p_split=0.20` could not communicate that setting to callers. |
| 4-bracket context ladder | Replaced the single 48 k-token cliff with a stepped policy derived from acceptance-rate evidence in `ROCmFPX-EXPERIMENT.md`: |

**`fp3-mtp` policy ladder (new):**

```
< 16 384 tokens  → n_max=4, p_min=0.75, p_split=0.10
16 384–49 151    → n_max=4, p_min=0.25, p_split=0.10
49 152–98 303    → n_max=2, p_min=0.0,  p_split=0.10
≥ 98 304         → n_max=1, p_min=0.0,  p_split=0.10
```

**`dense-coder` policy (new — context-aware):**
```
< 98 304 tokens  → n_max=6, p_min=0.0, p_split=0.20
≥ 98 304         → n_max=3, p_min=0.0, p_split=0.10  (backed off at extreme context)
```

**`fp4-general` policy (new — context-aware):**
```
< 98 304 tokens  → n_max=4, p_min=0.0, p_split=0.10
≥ 98 304         → n_max=2, p_min=0.0, p_split=0.10
```

---

### `ggml/rocmfpx/rocmfpx.c`

#### C1 — Binary search for `rocmfpx_nearest_scale_ue4m3`

The original implementation was an O(126) linear scan over the UE4M3 table. Replaced with a binary search (matching the existing `rocmfp4_nearest_scale_ue4m3` in `rocmfp4.c`). The UE4M3 table is monotonically increasing, so the binary search narrows to a 2-element window then picks the closer neighbour. Tie-breaking kept identical to the old scan (prefer the lower byte value).

Estimated impact: ~18× reduction in `nearest_scale` call cost. Called ~(n_params/16) times during quantization — meaningful on large models.

#### C2 — Precomputed `rocmfpx_scale_table[127]`

Added a static `const float rocmfpx_scale_table[127]` initialised at compile time with all valid UE4M3 → FP32 values. Added `static inline rocmfpx_scale_lookup(uint8_t e)` for O(1) table access. All internal uses of `rocmfpx_ue4m3_to_fp32()` inside this file (MSE inner loops, scale-search clip checks, quantize/dequantize row functions) replaced with `rocmfpx_scale_lookup()`. The public `rocmfpx_ue4m3_to_fp32()` function is preserved for external callers.

#### C3 — `all_finite` fast path for MSE inner loops

`rocmfpx_prepare_mse_weights()` gained a `bool * all_finite` output parameter. Six `_finite` variants of the three MSE inner-loop functions were added (one unweighted + one weighted per format: FP3, FP6, FP8). These variants skip the `isfinite(x[i])` guard per element. All three scale-search dispatch functions (`_choose_scale_fp3_mse_impl`, `_choose_scale_fp6_mse_impl`, `_choose_scale_fp8_weighted_mse`) now propagate `all_finite` and route to the appropriate variant. In the common case of normal model weights (no NaN/Inf), this eliminates one conditional branch per element per MSE candidate evaluation.

#### C4 — Group pack/unpack replacing bit-by-bit `set_bits`/`get_bits`

**Removed:** `rocmfpx_set_bits` and `rocmfpx_get_bits`. Both looped over individual bits with a branch + read-modify-write per bit.

**Added:** Four `static inline` group functions:

| Function | Input → Output | Elements/call |
|---|---|---|
| `rocmfpx_fp3_pack8(dst, codes)` | 8 × uint8 → 3 bytes | 8 |
| `rocmfpx_fp3_unpack8(src, codes)` | 3 bytes → 8 × uint8 | 8 |
| `rocmfpx_fp6_pack4(dst, codes)` | 4 × uint8 → 3 bytes | 4 |
| `rocmfpx_fp6_unpack4(src, codes)` | 3 bytes → 4 × uint8 | 4 |

Both formats use the natural 3-byte group size (lcm(3-bits, 8) = 24 bits; lcm(6-bits, 8) = 24 bits). Every output byte is fully determined by the pack expressions — the `memset(yb->qs, 0, …)` call that preceded the old loop was removed.

**Rewritten:** All 4 quantize-row and 2 dequantize-row functions for FP3/FP6. Quantize collects codes into a stack array then calls pack in groups; dequantize unpacks all codes upfront then loops over elements with no bit arithmetic.

FP3 layout (3 bytes per 8 elements):
```
byte 0: v0[2:0] | v1[2:0]<<3 | v2[1:0]<<6
byte 1: v2[2]   | v3[2:0]<<1 | v4[2:0]<<4 | v5[0]<<7
byte 2: v5[2:1] | v6[2:0]<<2 | v7[2:0]<<5
```

FP6 layout (3 bytes per 4 elements):
```
byte 0: v0[5:0] | v1[1:0]<<6
byte 1: v1[5:2] | v2[3:0]<<4
byte 2: v2[5:4] | v3[5:0]<<2
```

---

## Future sessions — append below this line

## Session 002 — 2026-06-21

**Scope:** ROCmFPX production-preflight, agent quant, imatrix, DFlash capability, TurboQuant, and serving polish.

### `scripts/rocmfpx-model-capabilities.py`

| Fix | Detail |
|-----|--------|
| Capability helper | Added lightweight GGUF capability detection for MTP, diffusion, QAT, agent/coherent, and DFlash/DDFlash markers. |
| Serving profiles | Added model-aware serving profiles for known Nemotron agent, Qwen/Qwable MTP, generic MTP, diffusion, QAT, agent, and regular GGUF cases. |
| False-positive guard | Restricted short `qat` / `diffusion` matches to filenames and longer metadata markers to avoid raw tensor byte false positives. |

### `scripts/rocmfpx-production-preflight.sh`

| Fix | Detail |
|-----|--------|
| Preflight JSON | Added model kind/capability fields, full `launch_command`, warnings/errors, and `WRAPPER_OUT` generation. |
| Safety gates | Added hard-fail options for `REQUIRE_MTP=1` and `REQUIRE_PROFILE=1`. |

### `scripts/run-rocmfpx-mtp-server.sh`

| Fix | Detail |
|-----|--------|
| MTP auto-detect | Added model capability check so non-MTP models do not receive invalid `draft-mtp` flags unless `REQUIRE_MTP=1`. |
| Utilization knobs | Added `PERF_PRESET`, `PARALLEL`, polling, priority, GPU-layer, FlashAttention, fit, split, and host/offload toggles. |

### `scripts/quantize-rocmfpx-agent.sh`

| Fix | Detail |
|-----|--------|
| Imatrix support | Added `IMATRIX=/path/to/imatrix.gguf` pass-through to `llama-quantize --imatrix` with missing-file validation. |

### `ggml/rocmfpx/rocmfpx.c`

| Fix | Detail |
|-----|--------|
| Imatrix scale search | Added weighted quantization paths for ROCmFP3, ROCmFP6, and ROCmFP8 so accepted imatrix data affects ROCmFPX scale selection. |

### `ggml/rocmfpx/test_rocmfpx.c`

| Fix | Detail |
|-----|--------|
| Imatrix tests | Added weighted-MSE checks proving imatrix improves FP3/FP6/FP8 reconstruction on targeted calibration-weight cases. |

### `src/llama-kv-cache.cpp`

| Fix | Detail |
|-----|--------|
| TurboQuant policy | Added opt-in boundary-layer K protection for symmetric TurboQuant cache experiments. |

### Docs and gates

| File | Detail |
|------|--------|
| `README.md` | Added ROCmFPX family, agent quant, imatrix, TurboQuant, and contributor guidance. |
| `docs/ROCmFPX-SERVING.md` | Added preflight, model-kind guidance, utilization knobs, imatrix usage, TurboQuant asymmetric KV, and DFlash safety notes. |
| `docs/ROCmFPX-EXPERIMENT.md` | Documented ROCmFPX imatrix reference coverage. |
| `docs/ROCmFPX-HANDOFF.md` | Added handoff notes for ROCmFPX family usage and safety boundaries. |
| `scripts/check-rocmfpx-model-capabilities.sh` | Added synthetic capability/preflight/wrapper smoke coverage. |
| `scripts/check-rocmfpx-all.sh`, `scripts/check-rocmfpx-summary.sh` | Added capability gate integration. |
| `scripts/run-rocmfpx-turboquant-asym-server.sh` | Added safe asymmetric TurboQuant K/V serving wrapper. |

### Validation

| Check | Result |
|-------|--------|
| `scripts/build-strix-rocmfp4-mtp.sh llama-quantize` | passed |
| `scripts/check-rocmfpx-model-capabilities.sh` | passed |
| `scripts/check-rocmfpx-reference.sh` | passed, including imatrix weighted FP3/FP6/FP8 checks |
| Python/shell syntax checks | passed |
| DiffusionGemma BF16 → ROCmFP4 coherent agent quant | passed, `13,764.94 MiB / 4.57 BPW` |

<!-- TEMPLATE FOR FUTURE AI SESSIONS:

## Session NNN — YYYY-MM-DD

**Scope:** Brief description of what was changed and why.

### `path/to/file.ext`

| Fix | Line(s) | Detail |
|-----|---------|--------|
| Short label | L123 | What changed and why. |

*(Repeat per file. Keep entries factual. Do not remove or rewrite earlier sessions.)*

-->

## Session 003 — 2026-08-14

**Scope:** Muse-Glimmer-30B architecture guide, DFlash speculative decoding contract, long-context ATEM stability analysis, and local machine runbook isolation.

### `docs/MUSE-GLIMMER-DFLASH-SERVING.md`

| Fix | Detail |
|-----|--------|
| Serving guide & stability analysis | Added de-identified documentation covering Muse Glimmer architecture (52 layers, 39 SWA local with RoPE + 13 global with NoPE), DFlash block diffusion drafter contract (`--spec-draft-n-max 15`), failure mode analysis on context over-extrapolation causing XML/ATEM tag corruption, two-channel reasoning budget management, and production serving configurations. |

### `.gitignore`

| Fix | Detail |
|-----|--------|
| Local runbook isolation | Added `/LOCAL-*.md` and `/LOCAL-MUSE-GLIMMER-NOTES.md` to ignore local-specific paths and hardware benchmark artifacts. |

---

## Session 004 — 2026-08-18

**Scope:** Qwen 3.8 27B hybrid SSM-Attention architecture characterization, MTP speculative decoding verification (65/65 blocks), `ROCmFP4_FAST` quantization pipeline with imatrix calibration, and Strix Halo (Ryzen AI MAX+ 395 / gfx1151) benchmark validation.

### `docs/QWEN38-ROCMFP4-SERVING.md`

| Fix | Detail |
|-----|--------|
| Serving guide & benchmark envelopes | Added documentation covering Qwen 3.8 27B hybrid SSM architecture (64 backbone layers: 16 Gated Attention + 48 Gated DeltaNet SSM + 1 MTP draft layer), MTP speculative decoding contract (`--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.60`), `ROCmFP4_FAST` 4.26 BPW quantization workflow from high-precision Q8_0 parent + imatrix, asymmetric TurboQuant KV cache scaling across 262K context, and empirical Strix Halo benchmarks (346 tok/s PP, 30–36 tok/s MTP TG). |

### Validation

| Check | Result |
|-------|--------|
| `llama-quantize` Ninja + ROCm Clang build | passed, linked `llama-quantize.exe` |
| `Qwen3.8-27B-Uncensored-Q8_0.gguf` + `imatrix.dat` $\to$ `ROCmFP4_FAST` | passed, `13,877.14 MiB / 4.26 BPW`, all 866 tensors & 65/65 blocks verified |
| `llama-bench` on Radeon 8060S (`gfx1151`) | `pp512`: **346.54 ± 9.98 tok/s**, `tg64`: **13.75 ± 0.05 tok/s** unassisted, **~30.5–36.0 tok/s** with MTP |

---

## Session 005 — 2026-08-18

**Scope:** Multi-runner live benchmark sweeps, KV Cache quantization impact on Prompt Processing (PP), context shift robustness standardization (`--context-shift -n-keep -1`), and external technical tracking (`strix-halo-guide`, `Lucebox`, AMD Day-0 whitepapers).

### `LOCAL-ENVIRONMENT-MEMORY.md`

| Fix | Detail |
|-----|--------|
| Standard Serving Directives | Codified golden rules for all model launchers: `-np 1`, `--context-shift -n-keep -1`, native context limits, unquantized F16 KV cache on 96GB VRAM, and MTP/DFlash speculation. |
| External references & Upstream tracking | Indexed `strix-halo-guide` (Ryzen AI MAX+ 395 empirical benchmark authority), `Lucebox` (DFlash/DSpark drafter hub), and AMD Day-0 client whitepapers. |

### Batch Launchers (`C:\models\*.bat`)

| File | Change |
|------|--------|
| `Qwen3.8-27B-Uncensored_rocm714_mtp_textonly.bat` | Added `--context-shift -n-keep -1` |
| `Qwen3.8-27B-Uncensored_rocm714_mtp_vision.bat` | Added `--context-shift -n-keep -1` |
| `Qwen3.8-27B-UD-Q6K_rocm714_mtp_textonly.bat` | Added `--context-shift -n-keep -1` |
| `Qwen3.8-27B-UD-Q6K_rocm714_mtp_vision.bat` | Added `--context-shift -n-keep -1` |
| `Muse-Glimmer-30B_rocm714_dflash_textonly.bat` | Added `--context-shift -n-keep -1` |
| `Muse-Glimmer-30B_rocm714_dflash.bat` | Added `--context-shift -n-keep -1` |
| `Muse-Glimmer-30B-Abliterated_rocm714_dflash_textonly.bat` | Added `--context-shift -n-keep -1` |
| `Muse-Glimmer-30B-Abliterated_rocm714_dflash.bat` | Added `--context-shift -n-keep -1` |

### Validation & Benchmark Insights

| Check | Result | Key Takeaway |
|-------|--------|--------------|
| Live 3-Runner Sweep (`Qwen3.8-27B Q6_K`) | ROCm 7.14: **323.3 tok/s PP / 9.54 tok/s TG / 18.82 tok/s MTP** vs Vulkan: **163.4 tok/s PP** | ROCm HIP provides 2× faster Prompt Processing on `gfx1151`. |
| KV Cache Quant Sweep (Local build) | F16 KV: **227.6 tok/s PP** vs F16/Turbo4: **211.3 tok/s PP** vs Q8/Turbo4: **59.7 tok/s PP** | KV cache quantization degrades PP speed due to on-the-fly dequantization overhead in compute-bound phase. Native F16 KV is optimal when VRAM is abundant. |

---

## Session 006 — 2026-08-18

**Scope:** Cross-repository ecosystem evaluation (`daimonionnn/amd-rocmfpx-for-win`, `hec-ovi/llama-vulkan-strix`, `hec-ovi/vllm-qwen`, `JeremiahM37/strix-halo-sglang`), tool-calling quality validation, Strix Halo memory bandwidth laws, and MTP tuning optimizations.

### `LOCAL-ENVIRONMENT-MEMORY.md`

| Fix | Detail |
|-----|--------|
| Windows 11 Peer Evidence | Added `daimonionnn/amd-rocmfpx-for-win` benchmarking on Strix Halo: `Q6_K` ranked #1 in tool-eval-bench (88.1 score) across 7 quants, confirming optimal accuracy-speed tradeoff over ROCmFP4 (82.7 score). |
| Bandwidth Laws & MTP | Recorded empirical physical constant ($t/s \times \text{GiB} \approx 198$) confirming decode memory bandwidth saturation, and MTP tuning (`--spec-draft-n-max 6`, `--spec-draft-p-min 0.00`). |
| Framework Comparison | Documented comparative metrics against vLLM (4.3 t/s) and SGLang (1.7 t/s single-stream) establishing `llama.cpp` + ROCm GGUF as the definitive client runtime. |
| Qwen 3.8 27B ROCm Matrix | Added `AIwork4me/Qwen3.8-27B-ROCm` 28-cell benchmark receipts on `gfx1151`, validating 33.9 GiB memory footprint at 262K context, MTP acceleration, and formal vLLM non-interactive ruling. |
| ROCmFP4 Architecture Lab | Added `kingjones30/strix-halo-quant-lab` empirical findings: ROCmFP4 accelerates large-active MoE models (+62% on Laguna) but harms MLA/hybrid-linear long-context models (-37%), and indexed 55 HuggingFace pre-quantized models. |

### Batch Launchers (`C:\models\*.bat`)

| File | Change |
|------|--------|
| `Qwen3.8-27B-Uncensored_rocm714_mtp_textonly.bat` | Added `--reasoning-preserve`, tuned MTP to `n_max 2` (~18.8 tok/s), cleaned sampling to `--temp 0.7 --top-p 0.95`. |
| `Qwen3.8-27B-Uncensored_rocm714_mtp_vision.bat` | Added `--image-min-tokens 1024` (VL grounding fix), `--reasoning-preserve`, tuned MTP to `n_max 2`, cleaned sampling. |
| `Qwen3.8-27B-UD-Q6K_rocm714_mtp_textonly.bat` | Tuned MTP to `n_max 2`, cleaned sampling parameters. |
| `Qwen3.8-27B-UD-Q6K_rocm714_mtp_vision.bat` | Added `--image-min-tokens 1024`, tuned MTP to `n_max 2`. |
| `Muse-Glimmer-30B*.bat` (4 files) | Standardized with `--context-shift --keep -1`. |

---

## Session 007 — 2026-08-19

**Scope:** Multimodal video input diagnosis, `libmtmd` container constraints, and video keyframe extraction tooling.

### `LOCAL-ENVIRONMENT-MEMORY.md`

| Fix | Detail |
|-----|--------|
| Multimodal Video Protocol | Documented `libmtmd` format boundaries: raw `.mp4` container uploads trigger task cancellation; VLM video understanding requires keyframe slicing (8-16 frames). |
| Video Tooling | Created `C:\models\Extract-Video-Frames.bat` (drag-and-drop ffmpeg frame extraction) and `C:\models\Ask-Video.py` (automated frame extraction + streaming multimodal chat). |

---

## Session 008 — 2026-08-20

**Scope:** Unified flagship launcher for Qwen 3.8 27B on Strix Halo (`C:\models\Qwen3.8-27B_StrixHalo_Ultimate.bat`), integrating MTP sweet spot tuning (draft n=5, p=0.00), multimodal zero-overhead vision adapter, full 256K context shift safety, and universal Codex/WebUI Jinja compatibility.

### `C:\models\Qwen3.8-27B_StrixHalo_Ultimate.bat`

| Fix | Detail |
|-----|--------|
| Flagship Launcher | Created unified all-in-one launcher combining Q6_K high-accuracy model selection, persistent vision projection (`--image-min-tokens 1024`), optimized MTP speculation (`--spec-draft-n-max 5 --spec-draft-p-min 0.00` targeting 22~26+ tok/s), `--context-shift --keep -1`, `-c 262144`, native F16 KV cache, and `qwen3.8_codex_compatible.jinja`. |

### `LOCAL-ENVIRONMENT-MEMORY.md`

| Fix | Detail |
|-----|--------|
| Launcher Registry | Registered `Qwen3.8-27B_StrixHalo_Ultimate.bat` as the primary recommended entrypoint in the local launcher matrix. |

**Correction (2026-08-21):** the `--spec-draft-n-max 5` figure above does not hold up. The same-day full MTP sweep (n_max ∈ {2..6}) measured n=5 at 13.47 tok/s vs n=2's 20.23 tok/s, and the later end-to-end 6-turn real-conversation test found that *no* MTP/FP4 config beats the official ROCm 7.14 binary running Q6_K + MTP n=2 (103s vs 186-192s for every alternative tried), because this repo's local WIP build discards prefix-cache every turn. Until that cache-invalidation bug is fixed, this launcher is not the recommended entrypoint — treat it as experimental and prefer the plain Q6_K + official-binary + MTP n=2 config for daily use.

---

## Session 009 — 2026-09-07

**Scope:** Cherry-picked two fixes from `ROCmFPX/ROCmFPX` main (the real upstream, not the stale `ciru-ai` remote) onto `feat/dflash2-rocm`, and added a new adversarial tool-calling test script. Both cherry-picks were verified by direct rebuild + bench/smoke test rather than trusted on commit message alone; one turned out to be a no-op on this box's actual hardware, corrected in the same session rather than left as an unverified win.

### `ggml/rocmfpx/rocmfpx_mmq_rdna3.cuh`, `ggml/rocmfpx/test_rocmfpx_mmq.cpp`, `ggml/src/ggml-cuda/mmq-config-rdna3.cuh`, `scripts/check-rocmfpx-reference.sh`, `src/llama-model-loader.cpp`

| Fix | Detail |
|-----|--------|
| Cherry-pick `9b443bb2a` (upstream PR #20) | Restores RDNA3 HIP MMQ kernel selection + loader `ftype` classification for `Q4_0_ROCMFP4`/`_FAST`/`Q4_0_ROCMI4`/`Q2_0`/`Q3_0`/`Q5_0`/`Q6_0`/`Q7_0`/`Q8_0_ROCMFPX`, dropped by an earlier per-architecture MMQ selector refactor (`0b5be7e4a`). Applied cleanly, zero conflicts against this branch's own ROCmFPX commits. |
| **Verified scope, same session**: this fix is a no-op for gfx1151. `ggml-cuda/mmq.cuh`'s dispatch checks `GGML_CUDA_CC_IS_RDNA3_5(cc)` before `GGML_CUDA_CC_IS_RDNA3(cc)`, and `common.cuh:100` defines `GGML_CUDA_CC_RDNA3_5` with the comment "AI 370, AI Max 395 laptops" — this exact chip. gfx1151 has always used the separate `ggml_cuda_mmq_get_config_rdna3_5()` table (`mmq-config-rdna3-5.cuh`), which already carried all 91 ROCmFPX `CASE` entries, unaffected by the regression this PR fixes (that regression only ever hit desktop RDNA3.0 / RX 7000 cards). Benched to confirm: STRIX_LEAN quant pp512/tg128 unchanged within noise before/after. Kept anyway — harmless, and useful if the release zip is ever run by someone on a desktop RDNA3 GPU — but not a performance lever for this box. | |

### `src/models/qwen4exp.cpp`

| Fix | Detail |
|-----|--------|
| Cherry-pick `09412af38` (upstream PR #28023) | Qwen3.8-Flash-Next (qwen4exp) QSA indexer head-reduction: replaced a transpose+`sum_rows` with adjacent-slice adds, and dropped a redundant `cont()` on the indexer query. Upstream author measured pp512 2170→2366 t/s (+9%) on an RTX PRO 6000 at 55K context, gain scaling with context depth — mechanism-relevant to this repo's own Flash-Next QSA depth-decay findings. Applied cleanly (auto-merge, no conflicts). Smoke-tested only (loads, correct output on our own HIP build) — real depth-scaling benefit not yet measured here, since the daily-driver Flash-Next bat runs the official Vulkan binary, not this branch's HIP-only build. |

### `scripts/server-test-adversarial-tc.py` (new)

| Fix | Detail |
|-----|--------|
| New test script | Adds the two tool-calling test classes the existing `server-test-function-call.py`/`server-test-parallel-tc.py` suites never covered: prompt injection via a tool result (fetched-webpage content tries to hijack the agent into a destructive delete + data-exfil email), and ambiguous tool selection by semantic content rather than surface keywords (a security-disclosure scenario where the wrong tool is the keyword-obvious one). Also adds `--filler-tokens N` to pad the conversation to a target depth, since no config's tool-calling had ever been checked past a fresh context slot. Run against the current daily driver (STRIX_LEAN ROCmFP4 + MTP n=3 strict-qwen): 2/2 at shallow depth, 2/2 again at a real measured ~73.8K prompt tokens. Single-case-per-class only — not a full battery; see `tool-calling-reliability-gap` notes for what's still open (no head-to-head against the Q6_K baseline yet, which matters because a third-party finding elsewhere reported ROCmFP4-class quants losing to Q6_K specifically on prompt-injection resistance). |





