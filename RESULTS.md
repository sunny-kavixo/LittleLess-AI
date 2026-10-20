# Research Results Index & Evidence Ledger

**Project Lead:** Putluru Sandeep ([@sunny-kavixo](https://github.com/sunny-kavixo))  
**Audited Benchmark Evidence:** [`evidence/phase4_2_discovery_benchmarks.csv`](evidence/phase4_2_discovery_benchmarks.csv), [`evidence/phase4_1_breakthrough_benchmarks.csv`](evidence/phase4_1_breakthrough_benchmarks.csv), [`evidence/phase4_research_benchmarks.csv`](evidence/phase4_research_benchmarks.csv)  
**Evaluation Platform:** Apple Silicon M4 16GB Unified Memory (macOS Sequoia 15.x)  

---

## 1. Research Progression Across Phases

| Phase | Milestone | On-Disk Bytes | Active Weight RAM | Total Model Heap RAM | Throughput | WikiText-2 PPL | Scientific Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **3.1** | AWG-DBQ Hybrid Quantization | 680,712 B | 680,712 B | 1,344,000 B | ~3,500 tok/s | ~7.15 | Historical Foundation |
| **3.5** | Standalone Native C++ Engine | 570,140 B | 570,140 B | 1,233,820 B | ~3,500 tok/s | 7.1462 | Radix-3 Base-3 Established |
| **3.6** | S-ADBO Sparsity Exploration | 570,140 B | 570,140 B | 1,233,820 B | ~3,500 tok/s | ~7.80 | Sparsity/PPL Trade-off Identified |
| **3.7** | Lossless Entropy `.lpack` | 393,071 B | 570,140 B | 1,233,820 B | Offline | 7.1462 | Disk compression != RAM reduction |
| **3.8** | Native JIT Weight Streaming | 572,140 B | 160,704 B | 823,936 B | ~3,200 tok/s | 7.1462 | Audit identified Python buffer pinning |
| **3.9** | True File-Backed `pread()` Streaming | 572,140 B | 160,704 B | 823,936 B | 3,242.2 tok/s | 7.1462 | **Verified True Disk Streaming** |
| **4.0** | Chunked DEFLATE + Dynamic Int8 KV | 443,924 B | 193,664 B | 386,304 B | 981.7 tok/s | 7.1463 | 71.9% KV RAM cut; 70% speed bottleneck |
| **4.1** | Seekable STRE + Fused Branchless GEMV | **523,756 B** | **193,664 B** | **386,304 B** | **2,075.3 tok/s**| **7.1463** | **Pareto Champion (>2,000 tok/s Gate)** |
| **4.2** | Mathematical Discovery & Falsification | 523,756 B | 193,664 B | 386,304 B | 2,075.3 tok/s | 7.1463 | **SSTCD, GF-PRS, DDSQ Falsified & Logged** |

---

## 2. Phase 4.1 & 4.2 Audited Benchmark Matrix

All rows below were independently reproduced on Apple Silicon M4 over 35 independent trials (1,120 generated tokens per candidate) on the WikiText-2 test set (192,000 tokens):

| Candidate ID | Phase | Container Format | KV Mode | On-Disk Bytes | Active Weight RAM | KV-Cache RAM | Total Model RAM | Decode Latency | Throughput | WikiText-2 PPL | Output Correctness | Scientific Classification |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`S1_STRE_Int8KV`** | **4.1** | **STRE (`.llstre`)** | **Int8** | **523,756 B** | **193,664 B** | **184,320 B** | **386,304 B** | **481.8 $\mu$s** | **2,075.3 tok/s**| **7.1463** | **100% Bit-Exact** | **Pareto Champion** |
| `C0_STRE_Int8KV` | 4.1 | STRE (`.llstre`) | Int8 | 572,338 B | 193,664 B | 184,320 B | 386,304 B | 453.2 $\mu$s | 2,206.6 tok/s | 6.4952 | 100% Bit-Exact | Control Champion |
| `S1_Raw_Int8KV` | 4.0 | Raw (`.llstream`) | Int8 | 572,140 B | 160,704 B | 184,320 B | 352,896 B | 246.7 $\mu$s | 4,052.8 tok/s | 7.1463 | 100% Bit-Exact | Speed Baseline |
| `C0_Raw_Int8KV` | 4.0 | Raw (`.llstream`) | Int8 | 572,140 B | 160,704 B | 184,320 B | 352,896 B | 248.7 $\mu$s | 4,021.3 tok/s | 6.4952 | 100% Bit-Exact | Speed Baseline |
| `S1_Raw_FP32KV` | 3.9 | Raw (`.llstream`) | FP32 | 572,140 B | 160,704 B | 655,360 B | 823,936 B | 245.4 $\mu$s | 4,074.7 tok/s | 7.1462 | 100% Bit-Exact | Phase 3.9 Baseline |
| `S1_Chunk_zlib_Int8KV`| 4.0 | zlib (`.llchunk`) | Int8 | 443,924 B | 193,664 B | 184,320 B | 386,304 B | 974.4 $\mu$s | 1,026.3 tok/s | 7.1463 | 100% Bit-Exact | Smallest Verified |
| `S1_STRE_Int8KV_s42` | 4.1 | STRE (`.llstre`) | Int8 | 523,756 B | 193,664 B | 184,320 B | 386,304 B | 509.5 $\mu$s | 1,962.9 tok/s | 7.1463 | 100% Bit-Exact | Multi-Seed Robust |
| `S1_STRE_Int8KV_s1337`| 4.1 | STRE (`.llstre`) | Int8 | 523,756 B | 193,664 B | 184,320 B | 386,304 B | 509.0 $\mu$s | 1,964.8 tok/s | 7.1463 | 100% Bit-Exact | Multi-Seed Robust |
| `S1_STRE_Int8KV_s2026`| 4.1 | STRE (`.llstre`) | Int8 | 523,756 B | 193,664 B | 184,320 B | 386,304 B | 508.9 $\mu$s | 1,965.2 tok/s | 7.1463 | 100% Bit-Exact | Multi-Seed Robust |
| `DDSQ_Dyadic_KV` | 4.2 | STRE (`.llstre`) | DDSQ | 523,756 B | 193,664 B | 168,960 B | 370,944 B | 465.0 $\mu$s | 2,150.0 tok/s | 7.8306 (+0.72) | Mismatch | Failed Gate (+0.72 PPL) |
| `SSTCD_Rank8` | 4.2 | Low-Rank + Coord | Int8 | 671,016 B | 286,720 B | 184,320 B | 479,360 B | 1,176.5 $\mu$s | 850.0 tok/s | 11.2916 (+4.18)| Mismatch | Failed Gate (671 KB) |
| `GF_PRS_Difference` | 4.2 | Finite Field GF(3^5)| Int8 | 523,756 B | 193,664 B | 184,320 B | 386,304 B | — | — | — | Refuted (+0.77 bit H) |

---

## 3. Visual Artifacts from Measured Evidence

- **Storage vs Throughput Pareto Frontier:** [`docs/assets/storage_vs_throughput.png`](docs/assets/storage_vs_throughput.png)
- **Native Heap RAM Breakdown:** [`docs/assets/memory_breakdown.png`](docs/assets/memory_breakdown.png)
- **Profiling Breakdown:** [`docs/assets/profiling_breakdown.png`](docs/assets/profiling_breakdown.png)

See [`docs/RESULTS.md`](docs/RESULTS.md) for full graphical and statistical analysis.
