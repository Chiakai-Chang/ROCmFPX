# Qwen 3.8 27B & ROCmFP4 Serving Guide, MTP Speculation, and Strix Halo Envelopes

This document summarizes the architectural characteristics, speculative decoding integration (MTP), quantization workflows, context scaling envelopes, and production serving guidelines for the **Qwen 3.8 27B** model family on AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S / gfx1151).

---

## 1. Architecture Overview

Qwen 3.8 27B is a ~27.32B parameter hybrid dense SSM-Attention causal language model featuring native Multi-Token Prediction (MTP) and multimodal vision capabilities.

### Key Specifications

| Parameter | Specification |
|---|---|
| **Architecture** | Hybrid Gated DeltaNet SSM + Gated Attention (`qwen35` GGUF architecture) |
| **Total Layers** | 64 backbone layers + 1 MTP draft layer (Layer 64 / `blk.64`) |
| **Layer Pattern** | $16 \times (3 \times \text{Gated DeltaNet SSM} \to 1 \times \text{Gated Attention})$ |
| **SSM Configuration** | State dimension 128, conv1d kernel size 4, 48 SSM heads |
| **Attention Configuration** | 16 Gated Attention layers, 32 Q heads, 4 KV heads (GQA 8:1), head dimension 256 |
| **KV Cache Footprint** | Only 16 layers produce KV cache $\to$ **64 KiB/token (F16)** |
| **Trained Context Length** | **262,144 tokens (256K native)** |
| **Native MTP Layer** | Fused `nextn` projection head with shared embedding/norm (`blk.64.nextn.*`) |

---

## 2. Speculative Decoding with MTP (Multi-Token Prediction)

Qwen 3.8 embeds a native single-layer Multi-Token Prediction (MTP) head trained jointly with the main backbone.

### MTP Execution Contract in llama.cpp

* **Integrated Layer**: Unlike external drafter models (e.g. EAGLE or speculative companion models), MTP uses the final hidden state of the backbone network (Block 64) with a dedicated projection matrix (`blk.64.nextn.eh_proj.weight`) to generate draft token candidates in a single memory sweep.
* **CLI Parameters**:
  * `--spec-type draft-mtp`
  * `--spec-draft-n-max 4` (or `6` for deep code/JSON speculation)
  * `--spec-draft-p-min 0.60`
* **Performance Impact on APU Memory Bus**:
  * On Strix Halo's 256-bit unified memory bus (~190–200 GB/s sustained read bandwidth):
    * **Raw Unassisted Decode (`Q4_0_ROCMFP4_FAST`):** **13.75 tok/s** (1 memory load per token)
    * **MTP Speculative Decode (`n_max 4` / `p_min 0.60`):** 🔥 **30.56 – 36.04 tok/s** (2.2× – 2.6× speedup)

---

## 3. Quantization Pipeline: `Q4_0_ROCMFP4_FAST`

`ROCmFP4_FAST` is an RDNA 3.5 hardware-aligned block quantization format custom-engineered in ROCmFPX.

### Technical Characteristics

1. **Vector Register Alignment**: Groups exactly 32 weights per shared scale factor, perfectly matching RDNA 3.5 Wave32 / Wave64 SIMD register strides.
2. **Precision Balance**:
   * Backbone FFN and Attention weight tensors $\to$ `Q4_0_ROCMFP4_FAST` (4.25 bpw)
   * MTP projection head (`blk.64.nextn.eh_proj`) $\to$ Pinned at **`Q8_0`** (ensuring sharp candidate logit distribution)
   * SSM `conv1d`, `ssm_a`, and LayerNorms $\to$ **`F32`**
3. **Re-quantization from High-Precision Parent**:
   * Source model: `Qwen3.8-27B-Uncensored-Q8_0.gguf` (29.0 GB, JonathanColetti Heretic Pareto ARA)
   * Calibration: `Qwen3.8-27B-Uncensored-imatrix.dat`
   * Target model: `Qwen3.8-27B-Uncensored-ROCmFP4-FAST.gguf` (**13.55 GiB / 4.26 BPW**)

```bash
# Compilation
cmake --build build-win-hip-ninja --target llama-quantize -j 16

# Quantization
llama-quantize \
  --allow-requantize \
  --imatrix Qwen3.8-27B-Uncensored-imatrix.dat \
  Qwen3.8-27B-Uncensored-Q8_0.gguf \
  Qwen3.8-27B-Uncensored-ROCmFP4-FAST.gguf \
  Q4_0_ROCMFP4_FAST 16
```

---

## 4. Context Scaling & TurboQuant KV Cache

Because only 16 of the 64 layers are attention layers, Qwen 3.8 exhibits exceptionally lightweight KV cache scaling:

| Context Window | Model Weights | KV Cache (F16) | TurboQuant KV (`-ctk q8_0 -ctv turbo4`) | Total RAM (TurboQuant) |
|---|---|---|---|---|
| **8,192 (8K)** | 13.55 GiB | 0.50 GiB | 0.62 GiB | **14.17 GiB** |
| **32,768 (32K)** | 13.55 GiB | 2.00 GiB | 2.45 GiB | **16.00 GiB** |
| **65,536 (64K)** | 13.55 GiB | 4.00 GiB | 4.90 GiB | **18.45 GiB** |
| **131,072 (128K)** | 13.55 GiB | 8.00 GiB | 9.80 GiB | **23.35 GiB** |
| **262,144 (256K)** | 13.55 GiB | 16.00 GiB | 20.08 GiB | **33.63 GiB** |

---

## 5. Benchmarks on AMD Strix Halo (Ryzen AI Max+ 395)

Measured on **ASUS ProArt PX13 / Ryzen AI MAX+ 395 (40 CU Radeon 8060S @ 2.9 GHz, 128 GB LPDDR5X-8000, ROCm 7.14 / ROCmFPX HIP)**:

| Optimization Profile | Model Size | Prefill (`pp512 @ d0`) | Prefill (`pp512 @ d8K`) | Unassisted Decode | MTP Speculative Decode |
|---|---|---|---|---|---|
| Stock `Q6_K` (Heretic) | 20.90 GiB | 304.38 ± 2.23 tok/s | 259.78 ± 1.04 tok/s | 9.41 ± 0.03 tok/s | 18.82 ± 0.12 tok/s |
| Unsloth `UD-Q6_K_XL` | 24.10 GiB | 328.29 ± 3.33 tok/s | 273.09 ± 4.60 tok/s | 8.43 ± 0.04 tok/s | 17.51 ± 0.10 tok/s |
| **`ROCmFP4_FAST` (imatrix)** | **13.55 GiB** | **346.54 ± 9.98 tok/s** | **284.84 ± 2.90 tok/s** | **13.75 ± 0.05 tok/s** | 🔥 **30.56 – 36.04 tok/s** |

---

## 6. Recommended Production Serving Configurations

### Baseline Server Command (MTP Speculative + 256K Context)

```bash
llama-server \
  -m Qwen3.8-27B-Uncensored-ROCmFP4-FAST.gguf \
  -ngl 999 \
  --no-mmap \
  -fa on \
  -c 262144 \
  -b 2048 \
  -ub 512 \
  -ctk q8_0 \
  -ctv turbo4 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-p-min 0.60 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --host 0.0.0.0 \
  --port 8080
```

### Multimodal Vision Addition

To enable vision capabilities (Qwen-VL format), add the vision projection adapter:

```bash
  --mmproj Qwen3.8-27B-Uncensored-vision-f16.gguf
```
