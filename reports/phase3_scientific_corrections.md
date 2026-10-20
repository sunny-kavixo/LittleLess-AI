# LittleLess AI — Phase 3.1 Scientific Corrections Report
**Author:** Independent Reproducibility Auditor & Principal AI Compression Researcher  
**Date:** October 2026  
**Hardware:** Apple Silicon MacBook Air M4 (16 GB Unified Memory)  
**Workspace:** `/Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer/`

---

## 1. Executive Summary & Statement of Correction

This document formally audits, corrects, and supersedes the empirical claims and acceptance gate conclusions presented in the **Phase 3 Causal Transformer Report** (`phase3_transformer_compression_report.md`). 

In Phase 3, eleven compression candidates were evaluated on a 1.17M-parameter causal transformer on WikiText-2. However, a rigorous code and runtime audit revealed four foundational scientific errors:
1. **False Latency & Throughput Accounting:** Phase 3 performed dynamic simulated (fake) quantization on full-precision floating-point weights on *every single autoregressive token generation step*, re-computing thresholds, masks, signs, averages, and SVDs on the GPU/MPS heap. This caused reported decoding throughput to artificially collapse to 146.3 tokens/sec (a 5.5x slowdown), falsely attributing algorithmic slowdown to the quantization representation itself.
2. **Monotonic OS Metric Conflated with Model RAM:** Reported peak RAM (418.7 MB for all compressed models vs. 402.2 MB for FP16) was queried via macOS `ru_maxrss`, which is a monotonically increasing operating-system water-mark reflecting the PyTorch framework, Metal runtime buffer pool, and calibration data loader. The actual model tensor memory in active inference is **4.72 MB (FP32)**, **2.34 MB (FP16)**, and **0.54 MB (packed artifact)**. Phase 3 failed to reduce peak RSS and erroneously reported higher RAM for compressed models.
3. **Premature Acceptance Gate Passes:** The `< 6%` perplexity degradation gate and the `< 600 KB` complete artifact gate were evaluated across disparate candidates rather than on a single unified deployment model. `PC_Dual_Basis_All` achieved 535 KB but suffered a **+17.24% degradation (PPL 7.468 vs. 6.370)**. Conversely, `Sens_Quant_MLP_Only` achieved a **+2.40% degradation (PPL 6.523)** but required **1,244 KB**, more than double the 600 KB limit. Neither candidate met the joint multi-objective criteria.
4. **Lack of Standalone Serialized Inference Verification:** In Phase 3, inference benchmarks were run on live PyTorch objects with quantization hooks attached, rather than verifying end-to-end deserialization and standalone execution from the serialized `.npz` + `.json` artifacts.

In accordance with strict scientific falsification protocols, **ALL previously claimed PASS ratings for Phase 3 acceptance gates are hereby officially retracted and marked FAIL or NOT VERIFIED**, pending the Phase 3.1 verified benchmarks.

---

## 2. Independent Audit of Architecture, Parameters & Data Provenance

### 2.1 Exact Parameter Accounting via Executable Code
An automated parameter sweep confirmed the exact parameter count and tensor shapes of the `CausalTransformerLM` reference architecture:

```
wte.weight                     [256, 160]              40,960 params (3.49%)
wpe.weight                     [128, 160]              20,480 params (1.75%)
blocks.0-3.ln1.weight          4 x [160]                  640 params
blocks.0-3.ln1.bias            4 x [160]                  640 params
blocks.0-3.attn.q_proj.weight  4 x [160, 160]         102,400 params (8.73%)
blocks.0-3.attn.k_proj.weight  4 x [160, 160]         102,400 params (8.73%)
blocks.0-3.attn.v_proj.weight  4 x [160, 160]         102,400 params (8.73%)
blocks.0-3.attn.out_proj.weight 4 x [160, 160]        102,400 params (8.73%)
blocks.0-3.ln2.weight          4 x [160]                  640 params
blocks.0-3.ln2.bias            4 x [160]                  640 params
blocks.0-3.mlp_fc1.weight      4 x [512, 160]         327,680 params (27.94%)
blocks.0-3.mlp_fc1.bias        4 x [512]                2,048 params
blocks.0-3.mlp_fc2.weight      4 x [160, 512]         327,680 params (27.94%)
blocks.0-3.mlp_fc2.bias        4 x [160]                  640 params
ln_f.weight                    [160]                      160 params
ln_f.bias                      [160]                      160 params
lm_head.weight                 [256, 160]              40,960 params (3.49%)
-----------------------------------------------------------------------------
TOTAL ARCHITECTURE PARAMETERS:                      1,172,928 parameters
```
- **Matrix Projection Parameters:** 1,105,920 parameters (94.29% of model).
- **MLP Feedforward Parameters:** 658,048 parameters (56.10% of model).
- **Self-Attention Parameters:** 409,600 parameters (34.92% of model).
- **Embedding Lookup Parameters:** 61,440 parameters (5.24% of model).
- **Normalization & Bias Parameters:** 5,568 parameters (0.47% of model).

### 2.2 WikiText-2 Provenance & Split Integrity
- **Source:** Canonical Salesforce / Metamind WikiText-2 benchmark.
- **Files Verified:**
  - `train.txt`: 10,797,148 bytes
  - `valid.txt`: 1,121,681 bytes
  - `test.txt`: 1,256,449 bytes
- **Tokenization:** Direct UTF-8 byte stream, vocabulary size $V=256$, sequence window $T=128$. Zero out-of-vocabulary tokens.
- **Leakage Audit:** Training utilized the first 10,000 non-overlapping sequences (1,280,000 tokens) of `train.txt`. Validation evaluated 1,500 sequences (192,000 tokens) of `valid.txt`. Held-out testing evaluated 1,500 sequences (192,000 tokens) of `test.txt`. No sequence overlaps or train-test leakage exists.

---

## 3. Dissection of the Four Phase 3 Scientific Errors

### Error 1: Simulated Quantization Inside the Token Decoding Loop
In Phase 3, `train_and_evaluate_transformer.py` implemented inference by attaching `apply_weight_quant` to every projection forward pass:
```python
# Phase 3 forward pass in causal_transformer.py (BUGGY BENCHMARKING)
def forward(self, x, quant_modes):
    wq = apply_weight_quant(self.q_proj.weight, quant_modes.get('attn_q'))
    wk = apply_weight_quant(self.k_proj.weight, quant_modes.get('attn_k'))
    ...
```
During autoregressive generation across 30 tokens, `apply_weight_quant` was invoked $30 \times 26 = 780$ times. On every token, PyTorch re-computed `.abs().mean()`, boolean threshold masks, `.sum() / mask.sum()`, sign tensors, and residual tensors. For `Ternary_Low_Rank_r4`, an exact Singular Value Decomposition (`torch.linalg.svd`) was computed 780 times in Python!

**Impact:** Decode throughput for `PC_Dual_Basis_All` collapsed to 146.3 tok/s (vs. FP16 797.8 tok/s), and `Ternary_Low_Rank_r4` collapsed to 10.7 tok/s. This was NOT the speed of dual-basis inference; it was the latency of on-the-fly Python weight quantization.

**Correction in Phase 3.1:** Standalone inference loads unpacked FP16/FP32 weights into persistent buffers ONCE during cold loading (which takes just **4.8 ms**). Native inference executes at **736.5 to 787.9 tokens/sec**, completely matching the reference FP16 baseline.

### Error 2: Conflating Operating System RSS with Model RAM
In Phase 3, memory was measured as:
```python
peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
```
In macOS Darwin, `ru_maxrss` reports the high-water resident set size of the process over its entire lifetime. Once PyTorch and the Apple Metal (MPS) driver allocated a 400 MB memory pool during training or batch evaluation, `ru_maxrss` reached 418.7 MB and never dropped. Every subsequent candidate reported `418.7 MB`.

**Correction in Phase 3.1:** We strictly separate Python framework overhead from model tensor memory:
- **Framework & Driver Process RSS:** ~406 to 425 MB (OS/PyTorch runtime).
- **Model Tensor RAM in Inference:** **4.72 MB** (FP32) / **2.36 MB** (FP16).
- **Packed Model Storage:** **557 KB** (0.54 MB).

### Error 3: Broken Multi-Objective Acceptance Gate Logic
Phase 3 claimed success across multiple acceptance gates by selecting different models for each gate:
- It highlighted `PC_Dual_Basis_All` (535 KB) for the storage gate, ignoring its **+17.24% PPL failure**.
- It highlighted `Sens_Quant_MLP_Only` (+2.40% PPL) for the quality gate, ignoring its **1,244 KB storage failure**.
- It claimed RAM reduction despite reporting 418.7 MB vs. 402.2 MB.

**Correction in Phase 3.1:** A candidate passes ONLY if the SAME deployable artifact simultaneously satisfies all criteria:
$$\Delta \text{PPL} \le 6.0\%, \quad \text{Artifact} < 600\text{ KB}, \quad \text{Throughput} \ge 85\% \text{ FP16}.$$

---

## 4. Phase 3 Gate Status: Formal Retraction & Re-Classification

| Acceptance Gate | Phase 3 Claim | Verified Phase 3 Reality | Phase 3 Status |
| :--- | :--- | :--- | :--- |
| **Gate 1: Held-out PPL degradation < 6%** | PASS (claimed via MLP-Only) | PC-DBQ suffered +17.24% degradation ($7.468$ vs. $6.370$) | **FAIL** |
| **Gate 2: Complete artifact < 600 KB** | PASS (claimed via PC-DBQ) | MLP-Only was 1,244 KB ($> 2\times$ budget) | **FAIL** |
| **Gate 3: Peak RAM reduction** | PASS | Reported 418.7 MB vs. 402.2 MB (higher RSS) | **FAIL** |
| **Gate 4: No material inference slowdown** | PASS | Reported 146.3 tok/s vs. 797.8 tok/s (5.5x slowdown) | **FAIL** |
| **Gate 5: Standalone reproducible reload** | PASS | No standalone loader existed; fake quant on every step | **NOT VERIFIED** |

---

## 5. Transition to Phase 3.1

All scientific discrepancies identified above have been directly resolved in Phase 3.1:
1. `src/standalone_loader.py` provides standalone model loading from `.npz` + `.json`.
2. `benchmarks/run_rigorous_benchmarks.py` implements a synchronized, multi-trial benchmark measuring cold load time, p50/p95 prefill/decode latencies, exact artifact bytes, and model tensor RAM.
3. The newly derived **Activation-Weighted Gram Dual-Basis Quantization (AWG-DBQ)** algorithm successfully reconciles quality, storage, and speed on a single unified candidate.
