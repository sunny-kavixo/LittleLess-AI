# LittleLess AI Research Roadmap

**Status:** Ongoing scientific research program. Phase numbers track experimental milestones and architectural generations.

| Priority | Workstream | Acceptance Criterion | Status |
|---|---|---|---|
| **P0** | **Phase 4.0 FFI Crash Hardening** | Centralize Python FFI bindings; explicitly define `argtypes` / `restype` on all native C++ entry points | **VERIFIED & RESOLVED** |
| **P0** | **Independent Reproducibility Suite** | Publish unified native C++ kernels, verification gates (`audit_phase3_9`, `audit_phase4_1`), and tests | **VERIFIED & INTEGRATED** |
| **P0** | **True File-Backed Weight Streaming** | POSIX `pread()` into bounded scratchpad with zero persistent packed-weight pinning in Python | **VERIFIED & COMPLETE** |
| **P1** | **Dynamic Int8 KV Cache** | Per-timestep dynamic scaling; verified 71.875% KV RAM reduction (655 KB $\rightarrow$ 184 KB at T=128) with bit-exact output match | **VERIFIED & COMPLETE** |
| **P1** | **Compressed Streaming Throughput** | Overcome chunk decompression bottleneck; achieve >2,000 tok/s via STRE Coalesced I/O | **VERIFIED & ACHIEVED** (2,043 - 2,172 tok/s) |
| **P1** | **Mathematical Discovery Track** | Rigorously test non-linear residue transforms, dyadic scales, and SVD coupling | **COMPLETED** (Hypotheses A, B, C falsified; empirical limits established) |
| **P2** | **Phase 5.0: Learned Vector Codebooks** | Replace linear SVD with non-linear vector codebooks over ternary residues | **PLANNED** |
| **P2** | **Phase 5.1: Apple Silicon Metal SIMD** | Hardware-accelerated ternary GEMV and bitplane decompression via Metal Shading Language | **PLANNED** |
| **P2** | **Phase 5.2: 1B+ Foundation LLM Transfer** | Validate multi-plane streaming and dynamic Int8 KV on open-weight LLaMA / Mistral architectures | **PLANNED** |

---

## Experimental Protocol & Scientific Ethics

1. **Hypothesis Formulation:** Every structural change begins with a mathematical hypothesis and clear falsification criteria.
2. **Deterministic Verification:** Exact token matching across independent seeds against PyTorch FP32 reference.
3. **Transparent Accounting:** Separate on-disk container bytes, active native heap allocations, KV cache memory, and process-level RSS.
4. **Honest Reporting:** Falsified hypotheses and negative results are published with full mathematical explanations.

---

## Publication Policy

All public releases must pass automated security audits (zero hardcoded secrets, zero private keys, zero giant binary dumps), unit test verification, and independent reproduction gates.
