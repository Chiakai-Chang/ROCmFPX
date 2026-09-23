# Qwen3.8-Flash-Next IQ2 (DS4 light pack) on Windows + ROCm, Radeon 8060S (gfx1151)

This folder documents how we run [dongnhdev/Qwen3.8-Flash-Next-OrcaUncensored-IQ2-Light](https://huggingface.co/dongnhdev/Qwen3.8-Flash-Next-OrcaUncensored-IQ2-Light) with this fork's `llama-server` on native Windows, AMD Strix Halo. The files are:

- `merge_ds4_ple.py`: turns the DS4 main file plus its PLE sidecar into one GGUF this fork can load.
- `run-flashnext-iq2.bat`: example launcher with the flags we measured.
- This README: the measurements behind those flags, and what has **not** been verified.

We do not redistribute the model or binaries. You download the original files and convert them yourself.

## Status in one paragraph

The model loads and runs correctly on ROCm. On our 6-turn agent workload it finished in **142.8 s**, against **193.6 s** for a dense Qwen3.8-27B IQ4_NL (the 193.6 s comes from an earlier batch; see caveats). Tool calling passed our 34-test gate. General reasoning quality has **not** been measured. The routed experts are 2-bit (IQ2_XXS gate/up, Q2_K down), so treat quality as unknown until you test it on your own tasks.

## Hardware and software used

- AMD Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB unified memory with a 64 GB GPU carve (64 GiB dedicated VRAM), Windows 11.
- The model stays on an NVMe drive. The PLE table is read from disk on demand; do not put it on slow storage.
- This fork, branch `flashnext-iq2-ds4`, built with the TheRock ROCm SDK clang (AMD clang 24). CMake: `GGML_HIP=ON`, `GGML_VULKAN=ON`, `GPU_TARGETS=gfx1151`, `CMAKE_BUILD_TYPE=Release`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`.

## Why a conversion is needed

The DS4 pack targets [antirez/ds4](https://github.com/antirez/ds4). Its Qwen3.8-Flash-Next graph supports Metal and CUDA only, and ds4 has no Windows build. The GGUF also differs from what this fork's `qwen4exp` loader expects:

| Difference | Handled by |
| --- | --- |
| The n-gram PLE table (`ple.weight`, Q4_1, 29.8 GiB) is in a separate sidecar file | `merge_ds4_ple.py` appends it as `per_layer_token_embd.weight` |
| `qwen4exp.rope.dimension_sections` is missing | the script adds `[11, 11, 10, 0]` (`mrope_section` in the official config.json) |
| `attention.compress_ratios` and `ple.layers` are u64 arrays | the script rewrites them as u32 |
| Q2_K down experts have their 640-wide input zero-padded to 768 | commit `14f2af2c5` accepts the padded shape and zero-pads the activation, which gives an exact result |

The script copies all tensor data byte for byte. Nothing is re-quantized.

## Steps

1. Build this branch (`flashnext-iq2-ds4`) with HIP for gfx1151.
2. Download two files from the dongnhdev repo (gated; accept the terms on HF first):
   - `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf` (43,677,591,168 bytes, sha256 `b1dd08509231126b5f1596603fd2589ff8bdf8eae221420f3e4409110fee8bd4`)
   - `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` (32,000,157,440 bytes)
3. Merge. It needs about 75.7 GB of free space and took 5.5 min on our NVMe:
   ```
   python merge_ds4_ple.py <main>.gguf Qwen3.8-Flash-Next-PLE-Q4_1.gguf Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-DenseQ4K-MTP-PLEmerged.gguf
   ```
   The output is 75,677,744,704 bytes. After checking that it loads, you can delete the two inputs.
4. Edit the paths in `run-flashnext-iq2.bat` and start it. The chat template is Sharp v22.5.0, a third-party template based on froggeric v22.5; it is not included here. The model's embedded template also works, with the caveats below.

## Kernel fix in this branch

Commit `d08094170` fixes the generic MMB prefill decode for IQ2_XXS and Q2_K. The old loop pinned each of the 8 lanes to one IQ2_XXS sub-block, so only 2 lanes wrote into the 64-wide slice and each did 32 values serially. Q2_K decoded all 256 values of a block to keep 64. Measured with `test-backend-ops` at T=4096 tokens, 512 experts, 10 used:

| op | before | after |
| --- | ---: | ---: |
| IQ2_XXS gate/up `MUL_MAT_ID` | 46.6 ms | 16.9 ms |
| Q2_K down, k=768 | 31.4 ms | 19.4 ms |

`MMB_QUANT` correctness tests pass (12/12, including the fused GLU path). Greedy output was byte-identical before and after. End to end, agent-workload prefill went from 489 to 636 t/s.

Other generic MMB tiers (for example Q3_K, Q5_K, Q6_K, IQ2_XS, IQ2_S, IQ3_*, IQ4_XS) look like they have the same lane pattern. They were not changed or measured.

## Measurements

The workload for rows marked "agent" is 6 turns: a 34K-character document, then about 3.4-4.1K new prompt tokens of tool output per turn and 400 generated tokens per turn. Sampling is temperature 0.60, top-k 20, top-p 0.95, repeat penalty 1.05. Each server was started fresh.

**Same-batch A/B, interleaved ABBA x2, agent workload** (embedded template, MTP off):

| config | session (s) | PP (t/s) | TG (t/s) |
| --- | --- | ---: | ---: |
| Qwen3.8-27B IQ4_NL, same fork, MTP n3 | 193.3 / 193.6 / 191.7 / 195.7 | 290.8 | 25.3 |
| Flash-Next IQ2 | 150.7 / 162.6 / 159.3 / 143.2 | 636.1 | 20.7 |

Two Flash-Next runs had one turn that stopped early (180 and 50 tokens). The two full-length runs average 160.9 s, 16.9% faster. Flash-Next wins on prefill and loses about 18% on decode. It helps most when each turn adds a lot of prompt, and it is slower for short-input, long-output use.

**MTP on the final template (Sharp v22.5.0), 2 balanced rounds, agent workload:**

| MTP | session (s) | TG (t/s) |
| --- | --- | ---: |
| off | 163.6 / 164.7 | 20.5 |
| n-max 1, p-min 0.40 | 150.4 / 148.7 | 24.0 |
| **n-max 2, p-min 0.40** | **142.8 / 142.8** | **25.5** |
| n-max 3, p-min 0.40 | 154.7 / 156.4 | 22.5 |
| n-max 2, p-min 0.00 | 145.7 / 141.0 | 25.7 |

Draft acceptance at n2 is 0.71-0.79. With the embedded template (default effort xhigh) at n3 / p-min 0.00, MTP had looked neutral. `--spec-mtp-strict-qwen` only applies to qwen35/qwen35moe and errors on qwen4exp.

**Other flags** (2 balanced rounds each):

- `-b/-ub`: 8192 gave PP 618 / 580, 4096 gave 579 / 563, 2048 gave 533 / 532. We keep 8192.
- Vulkan (same binary, `--device Vulkan0`): PP collapses to about 165 t/s. Do not use it for this file.
- `--ctx-checkpoints 16 --checkpoint-min-step 2048` against `8` / default: an edit-and-resend at 38K depth re-read 23 tokens in both, so we kept 8.
- Small requests (130-230 tokens) spend 1.1-1.4 s in prompt processing at shallow depth. This does not depend on `-ub`, MTP or `LLAMA_LAZY_READ_THREADS`, so it is a code-level cost.

**Prompt cache with the bat flags, grown to 136K tokens with 12K-token turns:**

- Each checkpoint is 112.6 MiB. There was never an `exceeds cache size limit` message.
- Switching to another conversation and back (A -> B -> A) at 136K re-read 29 tokens.
- Free system RAM stayed at 28-36 GiB.

**Memory:** at 262K context with MTP, 57.45 GiB dedicated VRAM and 4.0 GiB shared right after launch. After a 161-minute session at about 98K context: 59.39 GiB dedicated and 23.8 GiB free system RAM. The PLE page cache and the prompt cache compete for RAM, so watch it in long sessions.

**Tool calling** (`scripts/server-test-parallel-tc.py`, `server-test-function-call.py` in auto and forced mode, `server-test-adversarial-tc.py`): 12/12, 10/10, 10/10, 2/2, with both the embedded and the Sharp template.

**Real agent use:** one 161-minute Hermes Agent session, observations only:

- Within a conversation, 93-99.9% of the prompt was reused from cache each turn. A cancel followed by a resend re-read 4 tokens.
- Decode ran at 20-25 t/s at 20-50K depth and 16-19 t/s at 80-107K. Very predictable output reached 29 t/s.
- The fixed per-request prompt cost grows with depth: about 1.5 s at 20K and 2-2.8 s at 80-100K.

## Not verified, and caveats

- **General quality** (reasoning, knowledge, zh/en writing) was not measured against any baseline. Tool-calling tests passing does not mean the model is as capable as larger-bit quants.
- The model card itself notes that the 2-bit experts dilute the abliteration. Uncensored behavior is partial.
- The 142.8 s vs 193.6 s comparison is across batches. The same-batch A/B used the embedded template without MTP.
- Text only: the DS4 pack's vision tower was not set up.
- The embedded chat template returns an error for `reasoning_effort: "high"` (it accepts only xhigh, medium and low) and defaults to xhigh. Sharp v22.5.0 accepts `"high"` and defaults to medium. Sharp also scans system, developer and user messages for markers such as `<|think_low|>` and rewrites the system prompt when it finds one, which forces a full re-prefill. Tool-role messages are not scanned.
- The example binds to 127.0.0.1. Our own setup bound to 0.0.0.0 for a Docker client, and the server has no API key by default.
- **License:** the dongnhdev card says apache-2.0, while its base `ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2` says `qwen-community-1.0`. Check the terms that apply to you before using or sharing the merged file.
- Other people report higher Flash-Next numbers on the same chip (for example strix-alloy on native Windows). We have not compared against them, so these numbers are not a claim of best performance on this hardware.

## Reproducing the kernel measurement

`test-backend-ops` on this branch has two compile errors unrelated to this work (`init_mul_mat_id_ids`, `test_l2_norm_batch`). We patched them locally to build it and did not commit that patch. The perf shapes above are `MUL_MAT_ID` with `n_mats=512`, `n_used=10`, `n=4096`, `m=640, k=2560` (gate/up, broadcast input) and `m=2560, k=768` (down).
