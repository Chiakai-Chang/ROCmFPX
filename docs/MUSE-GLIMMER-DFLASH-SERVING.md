# Muse-Glimmer-30B & DFlash Serving Guide and Stability Analysis

This document summarizes architectural characteristics, speculative decoding integration (DFlash), long-context stability boundaries, and production serving guidelines for the **Muse-Glimmer-30B** model family in `llama.cpp`.

---

## 1. Architecture Overview

Muse-Glimmer-30B is a ~29.6B parameter dense causal language model with integrated multimodal perception capabilities, tailored for local autonomous agentic workloads.

### Key Specifications

| Parameter | Specification |
|---|---|
| **Architecture** | Dense Causal Transformer + Perception Encoder (ViT-G/14) |
| **Layers** | 52 total layers |
| **Attention Pattern** | `[Local, Local, Local, Global]` repeating (39 Local SWA + 13 Global) |
| **Sliding Window (SWA)** | 2,048 tokens on local layers |
| **Position Encoding** | RoPE ($\theta = 500,000$) on local layers only; Global layers use NoPE |
| **Attention Heads (Q / KV)** | 32 / 2 (GQA ratio 16:1, head dimension 128) |
| **FFN Structure** | SwiGLU, intermediate dimension 19,968 |
| **Trained Context Length** | **131,072 tokens (128K)** |
| **Native Tool Protocol** | ATEM XML schema (`<atem:function_calls>`, `<atem:invoke>`, `<atem:parameter>`) |

---

## 2. Speculative Decoding with DFlash

Muse Glimmer incorporates **DFlash** (Block Diffusion for Flash Speculative Decoding), a lightweight 5-layer companion drafter that predicts blocks of tokens in a single forward pass.

### DFlash Mechanism & llama.cpp Contract

* **Block Size**: The drafter is trained with a block size of 16 tokens (1 base target token + up to 15 draft tokens).
* **CLI Parameter**: In `llama.cpp`, the speculative decoding parameter **must be clamped to `--spec-draft-n-max 15`**.
  * Setting `--spec-draft-n-max 16` exceeds the trained block size ($16 - 1 = 15$) and can cause draft verification misalignments or runtime clamp warnings.
* **Shared Target Context**: DFlash injects intermediate hidden states into its attention layers; it must run under `llama-server` sharing the target model's context (`--spec-type draft-dflash`).
* **Drafter Quantization & Memory Bandwidth**:
  * An F16 drafter (~4.8 GB) re-reads weights on every draft block, creating significant VRAM bandwidth pressure on unified memory (APU/UMA) architectures.
  * A quantized drafter (such as Q4_K, ~1.5 GB) shares the identical tokenizer and hidden dimension while reducing memory bus traffic by ~3.2×, providing substantial throughput gains.

---

## 3. Failure Mode Analysis: Long-Context Tag Corruption

### Observed Symptom
In long-running agent loops (>60K accumulated context tokens), the model occasionally produces malformed XML/ATEM tool-call tags (e.g., corrupted closing tags like `</atem:...>` with non-ASCII fragments or truncated parameter values), causing downstream JSON/tool parsers to fail.

### Root Cause Decomposition

1. **Context Over-Extrapolation Beyond Trained Envelope**:
   - The model architecture relies on 39 SWA local layers with RoPE and 13 NoPE global layers, trained up to **131,072 tokens**.
   - Attempting to force larger context sizes (e.g. 262K via metadata override) without dedicated YaRN / RoPE frequency scaling creates extreme attention dispersion once context exceeds ~60K–70K tokens.
   - Attention degradation directly impairs the model's ability to maintain rigid grammatical syntax (such as closing XML tags for tool calls).

2. **Remediation**:
   - Strictly bound `-c` to the trained envelope: **`131072`** (or conservative operative windows such as `32768` / `65536`).
   - Remove any manual context length metadata overrides unless calibrated scaling parameters are explicitly configured.

---

## 4. Chat Templates, Reasoning Channel & Tool Protocol

### Two-Channel Generation
Muse Glimmer utilizes a dedicated reasoning channel:
- **Thinking**: Encapsulated in `<|start|>assistant to=self<|message|>...<|eom|>`. The server extracts this into `reasoning_content`.
- **Response / Tool Call**: Encapsulated in `<|start|>assistant to=user<|message|>...<|eot|>` or `<|start|>assistant to=<tool><|message|>...`.

### Critical Serving Guidelines

1. **Token Budget (`max_tokens`)**:
   - Muse Glimmer is a deep-thinking model. If the client budget (`max_tokens`) is small (e.g., $\le 1024$), the entire allocation can be consumed by the reasoning channel, resulting in an empty `content` field and `finish_reason: "length"`.
   - **Recommendation**: Set client `max_tokens >= 2048` (or `4096` for multi-step agent actions).

2. **Controlling Reasoning Depth**:
   - Adjust reasoning effort via the system prompt directive: `Reasoning strength: low | medium | high | xhigh`.
   - For interactive chat or fast tool invocation, `medium` or `low` provides swift responses while preserving tool accuracy.

3. **Template Preservation**:
   - Standard releases: Use the official template normalizing reasoning headers.
   - Fine-tuned / Abliterated variants: Preserve the baked-in GGUF Jinja template when custom personas or alignment modifications are embedded.

---

## 5. Recommended Production Serving Configurations

### Baseline Robust Command (Speculative DFlash + ROCm / CUDA)

```bash
llama-server \
  -m model-30B-Q4_K_M.gguf \
  -md dflash-drafter-Q4_K.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 15 \
  -ngl 999 -ngld 999 \
  -c 131072 \
  -fa on \
  -b 4096 -ub 2048 \
  --jinja \
  --load-mode none \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --host 0.0.0.0 --port 8080
```

### Multimodal Addition
To enable image/document understanding alongside DFlash:
```bash
  --mmproj mmproj-Q8_0.gguf
```

### Key Flags Summary

| Flag | Recommended Value | Rationale |
|---|---|---|
| `-c` | `131072` | Adheres strictly to native trained context limit |
| `--spec-type` | `draft-dflash` | Activates DFlash block diffusion speculative decoding |
| `--spec-draft-n-max` | `15` | Matches the trained block size ($16 - 1$) |
| `-fa` | `on` | Required for peak attention throughput and drafter verification speed |
| `-b / -ub` | `4096 / 2048` | Balances prompt prefill and verification batch throughput |
| `--temp / --top-p / --top-k` | `1.0 / 0.95 / 64` | Meta recommended sampling distribution |
