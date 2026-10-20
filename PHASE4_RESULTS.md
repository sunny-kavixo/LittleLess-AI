# Phase 4 — Empirical Research Results & Scientific Analysis

**Status:** Completed and independently verified. All source code, native C++ kernels, verification gates, and empirical CSV logs are integrated into this repository.

---

## 1. Executive Summary: The Phase 4 Evolution

Phase 4 transformed LittleLess AI from a single-plane uncompressed streaming prototype into a multi-plane, coalesced compressed inference engine:
- **Phase 4.0 (Chunk Streaming & Dynamic Int8 KV):** Successfully proved that dynamic per-timestep Int8 KV cache reduces KV memory by **71.875%** ($655,360 \rightarrow 184,320$ B) with bit-exact output equivalence. However, naive chunked DEFLATE decompression dropped decode speed by ~70% ($3,243 \rightarrow 961$ tok/s) due to redundant syscall overhead and plane thrashing.
- **Phase 4.1 (STRE Breakthrough):** Introduced **Spatio-Temporal Residue Encoding (STRE)** with **Coalesced Plane I/O**. Reading both ternary planes in a single `pread()` syscall recovered throughput to **2,043.4 – 2,172.4 tok/s** (>2.1× faster than Phase 4.0 chunked), while preserving the 22.4% disk savings on sparse models ($443,924$ B) and 71.9% KV memory reduction.
- **Phase 4.2 (Discovery & Falsification):** Rigorously tested three speculative mathematical hypotheses (SSTCD, GF-PRS, DDSQ). All three were falsified under empirical testing, establishing clear mathematical boundaries for ternary transformer compression.

---

## 2. Audited Phase 4 Benchmark Matrix

All metrics below are independently reproducible using `benchmarks/run_phase4_1_breakthrough_benchmarks.py` and `tests/audit_phase4_1_independent.py` on an Apple Silicon M4 system (WikiText-2, $T=128$, 192k tokens):

| Model Candidate | Compression Mode | Container Size (Bytes) | Native Active Weights (Bytes) | KV Cache RAM (Bytes) | Total Native Heap (Bytes) | Perplexity (PPL) | Decode Speed (Tokens/s) | Verification Status |
|:---|:---|---:|---:|---:|---:|---:|---:|:---:|
| **C0 (Dense)** | Raw Radix-3 Stream | 572,140 | 160,704 | 184,320 | 353,344 | 6.4952 | 3,235.1 | **VERIFIED** |
| **C0 (Dense)** | Chunked DEFLATE | 569,290 | 177,184 | 184,320 | 369,824 | 6.4952 | 871.6 | **VERIFIED** |
| **C0 (Dense)** | **STRE Coalesced** | **569,290** | **193,664** | **184,320** | **386,304** | **6.4952** | **2,172.4** | **VERIFIED** |
| **S1 (Sparse)** | Raw Radix-3 Stream | 572,140 | 160,704 | 184,320 | 353,344 | 7.1463 | 3,246.6 | **VERIFIED** |
| **S1 (Sparse)** | Chunked DEFLATE | 443,924 | 177,184 | 184,320 | 369,824 | 7.1463 | 958.9 | **VERIFIED** |
| **S1 (Sparse)** | **STRE Coalesced** | **443,924** | **193,664** | **184,320** | **386,304** | **7.1463** | **2,043.4** | **VERIFIED** |

*Note on Memory Accounting:*
- Native Active Weight RAM is the bounded buffer allocated by C++ to hold the current layer's weights.
- STRE Coalesced mode uses an extra 32,960 bytes of scratchpad memory ($193,664$ B vs. $160,704$ B) to hold both p1 and p2 planes in contiguous memory for 1-syscall reading.
- Total native model heap includes active weights, Int8 KV cache, and intermediate activation scratchpads. Whole-process Python RSS is ~397 MB.

---

## 3. Profiling the Speed Breakthrough

Detailed profiling (`benchmarks/profile_phase4_breakdown.py`) revealed the exact bottleneck:

```
Phase 4.0 Chunked (961 tok/s):
  [ 58.2% I/O Syscall Overhead ] [ 28.4% zlib Decompression ] [ 13.4% GEMV Math ]

Phase 4.1 STRE Coalesced (2,043 tok/s):
  [ 12.1% I/O Syscall Overhead ] [ 44.8% zlib Decompression ] [ 43.1% GEMV Math ]
```

By coalescing two separate `pread()` calls into a single contiguous block read, kernel context switches were cut by 50%, shifting execution back toward compute-bound efficiency.

---

## 4. Phase 4.2 Falsified Hypotheses

For complete mathematical definitions and empirical charts of the falsified Phase 4.2 experiments, see:
- [LIMITATIONS.md](LIMITATIONS.md)
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- [docs/MATHEMATICS.md](docs/MATHEMATICS.md)

| Hypothesis | Proposed Mechanism | Expected Benefit | Actual Empirical Outcome | Scientific Verdict |
|:---|:---|:---|:---|:---:|
| **Hypothesis A (SSTCD)** | SVD low-rank ($r=2$) + ternary residual bitplane | Break ternary limit; reduce size to <350 KB | Container expanded by +17.4% (671 KB); PPL degraded by +4.18 | **FALSIFIED** |
| **Hypothesis B (GF-PRS)** | Galois Field $\mathbb{F}_{3^5}$ polar non-linear transform | Compress ternary states into non-uniform distribution | Entropy increased from 7.00 to 7.78 bits/byte due to lack of metric continuity | **FALSIFIED** |
| **Hypothesis C (DDSQ)** | Pure dyadic power-of-two KV scaling ($2^k$) | Eliminate FP32 multiplier in attention | 1,100× MSE increase on keys; PPL degraded by +0.72 | **FALSIFIED** |

---

## 5. Next Steps (Phase 5.0)

With streaming I/O overhead resolved, future speedups must focus on SIMD decompression acceleration (NEON / Metal Shading Language) and non-linear learned vector codebooks. See [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md).
