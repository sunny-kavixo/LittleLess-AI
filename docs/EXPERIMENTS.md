# LittleLess AI — Experimental Ledger & Benchmark Methodology

**Author:** Putluru Sandeep ([@sunny-kavixo](https://github.com/sunny-kavixo))  
**Hardware Platform:** Apple Silicon M4 (16GB Unified Memory, macOS 15.x Darwin arm64)  
**Evaluation Dataset:** WikiText-2 Test Set (192,000 tokens, 1,500 batches, block size 128)  

---

## 1. Experimental Methodology & Rigor

All benchmarks adhere to strict independent scientific protocol:
1. **Isolated Execution:** Benchmarks run with zero background memory contention, isolated process PIDs, and bounded CPU cores.
2. **Warmup Protocol:** 5 full forward generation passes are executed before any timer starts to eliminate JIT, OS page-fault, and dynamic-linker cold-start noise.
3. **Statistical Robustness:** Every latency measurement is repeated across 35 independent trials (32 generated tokens per trial, generating 1,120 tokens per candidate). P50 median latencies are reported.
4. **Exact Determinism:** Every candidate must reproduce identical output tokens against the verified baseline sequence:
   - S1: `[105, 110, 32, 116, 104, 101, 32, 115, 101, 97]` (`"in the sea"`)
   - C0: `[32, 44, 32, 97, 110, 100, 32, 116, 104, 101]` (`" , and the"`)
5. **Multi-Seed Validation:** All conclusions are tested across 3 independent initialization seeds (42, 1337, 2026).

---

## 2. Comprehensive Experimental Ledger

The following ledger records all evaluated mechanisms from Phase 3.9 through Phase 4.2. Raw evidence is persisted in [`evidence/phase4_2_discovery_benchmarks.csv`](file:///Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer/evidence/phase4_2_discovery_benchmarks.csv).

| Experiment ID | Phase | Model | Container Format | KV Mode | On-Disk Bytes | Active Weight RAM | KV-Cache RAM | Decode Latency | Throughput | WikiText-2 PPL | Output Match | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `S1_Raw_FP32KV` | 3.9 | S1 | Raw (`.llstream`) | FP32 | 572,140 B | 160,704 B | 655,360 B | 245.4 $\mu$s | 4,074.7 tok/s | 7.1462 | Exact Match | Reproducible |
| `S1_Raw_Int8KV` | 4.0 | S1 | Raw (`.llstream`) | Int8 | 572,140 B | 160,704 B | 184,320 B | 246.7 $\mu$s | 4,052.8 tok/s | 7.1463 | Exact Match | Reproducible |
| `C0_Raw_Int8KV` | 4.0 | C0 | Raw (`.llstream`) | Int8 | 572,140 B | 160,704 B | 184,320 B | 248.7 $\mu$s | 4,021.3 tok/s | 6.4952 | Exact Match | Reproducible |
| `S1_Chunk_zlib_Int8KV`| 4.0 | S1 | DEFLATE (`.llchunk`) | Int8 | 443,924 B | 193,664 B | 184,320 B | 974.4 $\mu$s | 1,026.3 tok/s | 7.1463 | Exact Match | Reproducible |
| **`S1_STRE_Int8KV`** | **4.1** | **S1** | **STRE (`.llstre`)** | **Int8** | **523,756 B** | **193,664 B** | **184,320 B** | **481.8 $\mu$s** | **2,075.3 tok/s**| **7.1463** | **Exact Match**| **Champion** |
| **`C0_STRE_Int8KV`** | **4.1** | **C0** | **STRE (`.llstre`)** | **Int8** | **572,338 B** | **193,664 B** | **184,320 B** | **453.2 $\mu$s** | **2,206.6 tok/s**| **6.4952** | **Exact Match**| **Champion** |
| `S1_STRE_Int8KV_s42` | 4.1 | S1 | STRE (`.llstre`) | Int8 | 523,756 B | 193,664 B | 184,320 B | 509.5 $\mu$s | 1,962.9 tok/s | 7.1463 | Exact Match | Robust (Seed 42) |
| `S1_STRE_Int8KV_s1337`| 4.1 | S1 | STRE (`.llstre`) | Int8 | 523,756 B | 193,664 B | 184,320 B | 509.0 $\mu$s | 1,964.8 tok/s | 7.1463 | Exact Match | Robust (Seed 1337) |
| `S1_STRE_Int8KV_s2026`| 4.1 | S1 | STRE (`.llstre`) | Int8 | 523,756 B | 193,664 B | 184,320 B | 508.9 $\mu$s | 1,965.2 tok/s | 7.1463 | Exact Match | Robust (Seed 2026) |
| `SSTCD_Rank4` | 4.2 | S1 | Low-Rank + Coord | Int8 | 633,527 B | 245,760 B | 184,320 B | 1,176.5 $\mu$s | 850.0 tok/s | 14.8210 (+7.71) | Mismatch | Failed |
| `SSTCD_Rank8` | 4.2 | S1 | Low-Rank + Coord | Int8 | 671,016 B | 286,720 B | 184,320 B | 1,176.5 $\mu$s | 850.0 tok/s | 11.2916 (+4.18) | Mismatch | Failed |
| `SSTCD_Rank16` | 4.2 | S1 | Low-Rank + Coord | Int8 | 753,284 B | 368,640 B | 184,320 B | 1,250.0 $\mu$s | 800.0 tok/s | 9.4520 (+2.34) | Mismatch | Failed |
| `GF_PRS_Difference` | 4.2 | S1 | Galois Field Difference | Int8 | 523,756 B | 193,664 B | 184,320 B | — | — | 7.1463 | — | Refuted (+0.77 bit H) |
| `DDSQ_Dyadic_KV` | 4.2 | S1 | STRE (`.llstre`) | DDSQ | 523,756 B | 193,664 B | 168,960 B | 465.0 $\mu$s | 2,150.0 tok/s | 7.8306 (+0.72) | Mismatch | Failed (+0.72 PPL) |

---

## 3. Visual Artifacts from Measured Evidence

### Storage vs Throughput Pareto Analysis
![Storage vs Throughput](assets/storage_vs_throughput.png)

### Native C++ Model Heap RAM Footprint
![Memory Breakdown](assets/memory_breakdown.png)

### Profiling Breakdown per Forward Token Step
![Profiling Breakdown](assets/profiling_breakdown.png)
