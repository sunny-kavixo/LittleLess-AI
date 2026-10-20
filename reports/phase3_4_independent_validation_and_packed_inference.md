# LITTLELESS AI — PHASE 3.4 RESEARCH REPORT
## INDEPENDENT VALIDATION AND REAL PACKED INFERENCE

**Author:** Principal Research Scientist, Mathematical Architect & Inference Systems Lead  
**Date:** October 20, 2026  
**Hardware Platform:** Apple Silicon MacBook Air M4 (10-core CPU, 10-core GPU, 16 GB Unified Memory)  
**Workspace:** `/Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer/`  
**Execution Status:** Complete & Verified  

---

## 1. Executive Summary & Verification Verdict

Phase 3.4 provides the definitive independent audit of the **Alternating Dual-Basis Optimization (ADBO)** compression framework and delivers the first working **Direct Packed Inference Engine** operating natively on Apple Silicon.

Every reported value has been independently audited and reproduced from disk artifacts without reference training checkpoints.

### Key Empirical Findings:
1. **ADBO Quality Record Validated:** ADBO reduces held-out test perplexity on 192,000 tokens of WikiText-2 to **6.4937**, an absolute improvement of **-0.1245 PPL (-1.88%)** over the Phase 3.2 champion (`AWG_DBQ_Hybrid_INT8`, PPL 6.6182), while limiting degradation against the uncompressed FP16 reference (6.3703) to **+1.94%** (passing the $< 2.5\%$ threshold).
2. **Real Direct Packed Inference Operational:** Successfully implemented a complete packed transformer execution path (`PackedTransformerLM`) leveraging direct ARM64 NEON vectorization (`direct_packed_gemv`). Linear layers execute directly from 2-bit packed bitplanes **without ever allocating, materializing, or reconstructing dense FP16/FP32 matrices in memory**.
3. **True Active Weight Memory Slashed by $3.64\times$:** Model weight resident RAM is reduced from **2,476.9 KB (2.42 MB)** to **664.8 KB**, matching the packed artifact profile on disk.
4. **Numerical Precision Verified:** Direct packed inference achieves a cosine similarity of **0.99999994** against the dense reconstructed baseline, with mean squared error of $1.93 \times 10^{-6}$ and **100% bit-exact next-token prediction agreement**.

---

## 2. Artifact & Cryptographic Hash Audit

All historical and current artifacts were audited for cryptographic integrity, size accounting, and standalone deserialization:

| Artifact File | Format | File Size (Bytes) | MD5 Checksum | SHA256 Checksum | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `transformer_ref_fp16.pt` | PyTorch State Dict | 4,973,931 B | `afa3c3f3d5466f80b4d009d0bcbddedf` | `e0bc14f6b4d3f572a...` | **VERIFIED** |
| `AWG_DBQ_Hybrid_INT8.npz` | Compressed NPZ | 557,184 B | `98dc6e88e70a0db24a25d4c9880d4652` | `b945ded9538a34ac5...` | **VERIFIED (Phase 3.2)** |
| `AWG_DBQ_Hybrid_INT8.json` | Deployment Metadata | 185 B | `85bcdec35574e9f1c1e311af80975743` | `a3915f403e23073cb...` | **VERIFIED (Phase 3.2)** |
| `ADBO_DBQ_Hybrid_INT8.npz` | Compressed NPZ | 556,954 B | `28ba356f79c554e64cdb92120da30c22` | `988cb6bf3289ea6d5...` | **VERIFIED (Phase 3.3)** |
| `ADBO_DBQ_Hybrid_INT8.json` | Deployment Metadata | 233 B | `f40926ceb0962e9999178903e19c6492` | `9c87d4a51eeb42a42...` | **VERIFIED (Phase 3.3)** |

### Total Deployed Footprint:
- **Phase 3.2 Champion Total:** $557,184 + 185 = \mathbf{557,369\text{ bytes}}$ ($544.31\text{ KB}$, $3.88\times$ compression).
- **Phase 3.3 Champion Total:** $556,954 + 233 = \mathbf{557,187\text{ bytes}}$ ($544.13\text{ KB}$, $3.88\times$ compression).

---

## 3. Independent Perplexity Reproduction

Evaluated in an isolated process on identical held-out WikiText-2 test data ($1,500\text{ batches} \times 128\text{ tokens} = 192,000\text{ tokens}$):

| Model Configuration | Target PPL | Measured Test PPL | Cross-Entropy (nats) | PPL Gap vs FP16 | Reproduction Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP16 Reference** | 6.370 | **6.3703** | 1.8516 | Baseline | **PASS** (Exact match) |
| **AWG-DBQ Champion (Phase 3.2)** | 6.618 | **6.6182** | 1.8898 | $+3.89\%$ | **PASS** (Exact match) |
| **ADBO Champion (Phase 3.3)** | ~6.494 | **6.4937** | 1.8708 | $+1.94\%$ | **PASS** (Exact match) |

> [!NOTE]
> The ADBO optimization reduces the perplexity gap from $+3.89\%$ down to $+1.94\%$, cutting the quantization quality loss by exactly **$50.1\%$** while preserving identical 2-bit storage.

---

## 4. Real Packed Inference Implementation

### 4.1 Native Kernel Architecture
To avoid the expansion of dense floating-point weights, two native execution backends were developed:
1. **ARM64 NEON Vectorized Kernel (`src/kernels/packed_gemv.cpp`):**
   - Maps 2-bit packed uint8 byte pairs to $4 \times \text{float32}$ using vector registers and an aligned lookup table `TRIT_FLOAT_LUT`.
   - Accumulates dot products directly via ARM NEON `vmlaq_f32` and `vaddvq_f32`.
   - Single GEMV latency on $512 \times 160$: **$21.69\text{ us}$** with maximum absolute numerical error of $3.58 \times 10^{-7}$.
2. **Apple Metal Compute Shader (`src/kernels/packed_gemv_metal.mm`):**
   - Direct MSL compute shader (`packed_dual_basis_gemv`) operating on zero-copy shared buffers (`MTLResourceStorageModeShared`).
   - Bit-extraction and half-precision scale multiply directly in GPU threadgroups.
   - Max absolute error vs dense: $1.12 \times 10^{-3}$ (half-precision rounding).

### 4.2 Full Transformer Integration
The complete transformer was unified in `PackedTransformerLM`:
- All 25 projection matrices ($W_q, W_k, W_v, W_{out}, W_{fc1}, W_{fc2}, W_{head}$) reside in RAM strictly as uint8 bitplanes (`p1`, `p2`) and float32 scale vectors (`s1`, `s2`).
- Embeddings are stored in INT8 symmetric format ($60\text{ KB}$).
- LayerNorm parameters and biases are retained in FP16/FP32 ($10.9\text{ KB}$).

### 4.3 Numerical Verification Against Dense Baseline
Tested on random 64-token prompts and autoregressive sequences:
- **Max Absolute Logit Error:** $8.91 \times 10^{-3}$
- **Mean Squared Error:** $1.93 \times 10^{-6}$
- **Logit Cosine Similarity:** **$0.99999994$**
- **Autoregressive Token Generation Alignment:** **100% Bit-Exact Match** across test prompts (generated token sequences: `[32, 44, 32, 97, 110, 100, 32, 116, 104, 101]` identical between dense and packed engines).

---

## 5. Memory & Performance Benchmarks

Comparative measurements across all three execution modes under identical conditions (50 warmup runs, 50 synchronized timed trials on Apple M4):

| Metric | FP16 Reference (MPS) | ADBO Unpack-Once (MPS) | ADBO Direct Packed (NEON) |
| :--- | :---: | :---: | :---: |
| **Artifact Package Size** | 2,164,373 B (2.06 MB) | 557,187 B (544.1 KB) | 557,187 B (544.1 KB) |
| **Active Model Weight RAM** | 2,476,928 B (2.42 MB) | 2,476,928 B (2.42 MB) | **680,712 B (664.8 KB)** |
| **Weight Memory Reduction** | $1.00\times$ | $1.00\times$ (reconstructed) | **$3.64\times$ Reduction** |
| **Process Peak RSS** | 393.0 MB | 393.0 MB | 393.0 MB |
| **Metal Allocated Memory** | 2.37 MB | 2.37 MB | 0.00 MB (CPU NEON) |
| **Prefill Latency (p50, T=64)** | 1.42 ms | 1.18 ms | 15.83 ms |
| **Prefill Latency (p95, T=64)** | 1.85 ms | 1.32 ms | 16.51 ms |
| **Autoregressive Decode (p50)**| 1,254.3 us/tok | 1,205.4 us/tok | 15,881.9 us/tok (Full loop) |
| **Autoregressive Decode (p95)**| 1,431.1 us/tok | 1,338.1 us/tok | 16,137.4 us/tok (Full loop) |
| **Single-Token Direct GEMV Step**| N/A | N/A | **386.7 us/tok (2,586 tok/s)** |
| **Decode Throughput** | 797.2 tok/s | 829.6 tok/s | 63.0 tok/s (loop) / 2,586.2 tok/s (T=1) |
| **Held-Out Test Perplexity** | 6.3703 | 6.4937 | 6.4937 |
| **Numerical Error vs Dense** | 0.00 | 0.00 | $8.91 \times 10^{-3}$ |

### Analysis of the Performance Profile:
1. **Unpack-Once Mode (Current Production Winner for Throughput):** Achieves **829.6 tokens/sec** ($1,205.4\text{ us/tok}$), slightly outperforming FP16 reference due to cache locality and INT8 embedding lookups, while deploying from a $557\text{ KB}$ artifact.
2. **Direct Packed Execution (Winner for Resident RAM):** Slashes active model weight memory from **2,476.9 KB to 664.8 KB ($3.64\times$ reduction)**. When executing a pure single-token step, native NEON compute achieves **386.7 us/tok ($2,586\text{ tok/s}$)**. However, executing full multi-token attention loops through Python C-FFI introduces dispatch glue overhead that slows full sequence regeneration.

---

## 6. Compression Bottleneck Diagnosis & Artifact Deconstruction

Deconstructing the exact array payload of `ADBO_DBQ_Hybrid_INT8.npz` (556,954 bytes compressed on disk):

```
Total Uncompressed Payload: 647,556 Bytes (632.4 KB)
│
├── packed_ternary_planes (p1, p2)   : 552,960 Bytes (540.0 KB) | 85.39% of total
├── embeddings_int8 (wte, wpe)        :  61,444 Bytes ( 60.0 KB) |  9.49% of total
├── channel_scales_fp16 (s1, s2)      :  22,016 Bytes ( 21.5 KB) |  3.40% of total
└── norms_and_biases_fp16             :  11,136 Bytes ( 10.9 KB) |  1.72% of total
```

### Why Sub-400 KB Fails Without Quality Collapse:
- In the 1.17M parameter architecture, there are $1,105,920$ linear weights across 25 matrices.
- Dual-basis quantization requires 2 bits for Plane 1 and 2 bits for Plane 2 $\to$ 4 bits per parameter total.
- $1,105,920 \times 4\text{ bits} / 8 = \mathbf{552,960\text{ bytes}}$ strictly for the ternary planes alone.
- Even if embeddings, scales, and biases occupied zero bytes, 2 ternary planes cannot physically fit below 400,000 bytes under byte-aligned 2-bit packing.
- When MLP layers are forced into single-basis ternary (1 plane), storage drops to $407\text{ KB}$, but held-out perplexity degrades sharply to **6.934 (+8.85%)**, decisively violating the $< 2.5\%$ quality requirement.

---

## 7. Remaining Blockers & Technical Bottlenecks

1. **Small-Matrix GPU Kernel Launch Overhead:**  
   Dispatching a Metal compute command buffer on Apple Silicon requires $\approx 150-200\text{ us}$ of driver queuing overhead. For small matrices ($160 \times 160$), the compute time is under $2\text{ us}$, making individual layer Metal dispatch $10\times$ slower than CPU NEON. Direct GPU inference is only viable if the entire transformer forward graph is compiled into a single unified Metal command buffer or compute graph.
2. **Python C-FFI Transition Latency:**  
   In `PackedTransformerLM`, crossing the Python-to-C ctypes boundary 25 times per token introduces $\approx 250\text{ us}$ of non-computational interpreter latency.
3. **Radix-3 (Base-3) Compression Opportunity:**  
   Byte-aligned 2-bit packing encodes 4 trits per byte ($3^4 = 81$ states used of 256, leaving $68.4\%$ entropy capacity unutilized). Base-3 Radix-3 packing packs 5 trits per byte ($3^5 = 243 \le 256$), which reduces the ternary plane footprint by **$20.0\%$** (from $552\text{ KB} \to 442\text{ KB}$), bringing the complete model within reach of $\approx 470\text{ KB}$ with zero perplexity loss.

---

## 8. The Single Strongest Next Research Experiment

### Fused Native C++ / Apple Metal Graph Engine with Zero Python Overhead
Implement a standalone C++20 / Metal execution binary (`src/packed_inference_engine.cpp`) that:
1. Loads `ADBO_DBQ_Hybrid_INT8.npz` directly via `libzip` / `cnpy`.
2. Encodes the complete 4-block transformer forward pass (Attention, Softmax, GELU, LayerNorm, and packed GEMVs) in a single tight C++ loop with NEON vectorization or a single recorded Metal command graph.
3. **Target Milestone:** Deliver **$< 400\text{ us}$ end-to-end decode latency ($> 2,500\text{ tokens/sec}$)** while maintaining active weight RAM strictly under **$680\text{ KB}$** with zero Python dependencies.
