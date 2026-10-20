# Limitations, Negative Results, and Falsified Hypotheses

This document records scientific limitations, trade-offs, and falsified hypotheses rather than treating every experimental hypothesis as a success.

---

## 1. Quality vs. Compression Trade-offs

- **Sparsity vs. Quality Degradation:** Phase 3.6 S-ADBO achieved 70.94% zero-trit sparsity, but caused measurable perplexity degradation ($7.146$ PPL for sparse S1 vs. $6.495$ PPL for dense C0). Increasing sparsity does not guarantee model intelligence preservation.
- **Controlled Evaluation:** Results across different datasets, checkpoints, seeds, or precision modes must not be conflated without controlled baselines.

---

## 2. Storage vs. Active Memory Accounting

- **Disk Footprint vs. In-Memory Working Set:** Phase 3.7's 393 KB `.lpack` artifact is an on-disk entropy-coded format, requiring full memory expansion during decoding.
- **Native Active Allocations vs. Process RSS:** Native active-weight allocation (160 KB raw, 193 KB STRE) and KV cache (184 KB Int8) are bounded C++ heap allocations. Total Python process RSS (~397 MB) includes Python interpreter overhead, PyTorch runtime libraries, memory-mapped shared libraries, and OS file cache pages. These must never be conflated.
- **Transient Coalesced Buffer Trade-off:** Phase 4.1 STRE recovers throughput to >2,000 tok/s by coalescing plane I/O into a single `pread()` syscall, but requires 32,960 bytes of additional transient scratchpad memory ($193,664$ B vs. $160,704$ B).

---

## 3. Phase 4.2 Falsified Hypotheses & Negative Results

Scientific progress requires rigorous reporting of what did **not** work. In Phase 4.2, three speculative mathematical hypotheses were formulated and rigorously falsified:

### Hypothesis A: Structured SVD-Ternary Coupled Decomposition (SSTCD) — FALSIFIED
- **Hypothesis:** Weight matrices could be decomposed into low-rank SVD factors ($U V^T$, rank-2) plus sparse ternary residues ($T$) to boost compression beyond ternary bounds.
- **Finding:** **FAILED.** Storing 32-bit floating-point SVD components ($U, V$) alongside dense ternary residues ($45.02\%$ non-zero) introduced severe metadata overhead. Total container size **increased by +17.4%** ($572$ KB $\rightarrow 671$ KB), and language perplexity degraded severely by **+4.18 PPL** ($6.495 \rightarrow 10.677$). Low-rank factoring cannot compensate for residue entropy at this model scale.

### Hypothesis B: Galois Field Polar Residual System ($\mathbb{F}_{3^5}$ GF-PRS) — FALSIFIED
- **Hypothesis:** Mapping 5-trit blocks onto finite Galois Field $\mathbb{F}_{243}$ followed by non-linear polar transform would concentrate entropy into high-order coordinates.
- **Finding:** **REFUTED.** Finite field arithmetic over $\mathbb{F}_{3^5}$ lacks metric continuity. Modular wrapping scrambled the natural distribution of residual weights, **increasing Shannon entropy from 7.00 bits/byte to 7.78 bits/byte**, worsening downstream arithmetic compression efficiency.

### Hypothesis C: Dyadic Dynamic Scale Quantization (DDSQ) — FALSIFIED IN PURE FORM
- **Hypothesis:** Restricting scale factors strictly to dyadic powers of two ($2^k$) would eliminate FP32 dequantization multiplication in attention kernels.
- **Finding:** **FAILED IN PURE FORM.** Dyadic powers of two have an inherent 100% gap between successive steps ($2^k$ vs. $2^{k+1}$). In dynamic KV cache scaling, this coarse resolution produced a **1,100× increase in Mean Squared Error (MSE)** on key projections ($1.07 \times 10^{-4} \rightarrow 0.118$) and degraded perplexity by **+0.72 PPL**. Hardware floating-point multiplication remains necessary for KV scale factors.

---

## 4. Scale and Architectural Boundaries

- **Model Scale:** Current verified findings are established on a small experimental 6-layer causal transformer. Scalability to multi-billion parameter foundation LLMs is an open research question requiring future validation.
- **Novelty Boundaries:** Radix-3 packing, rANS, dynamic quantization, and POSIX `pread()` are established techniques. LittleLess AI's contribution is the specific mathematical co-design of multi-plane streaming containers and bounded native execution. We do not claim an unproven mathematical revolution.

---

## 5. Evidence Standards

- **Verified:** Independently reproduced with automated deterministic tests and inspectable CSV/log records.
- **Preliminary:** Measured in single-pass exploratory benchmarks awaiting multi-seed audit.
- **Falsified:** Empirically contradicted or shown to introduce net regressions.
- **Planned:** Proposed research direction not yet implemented.
