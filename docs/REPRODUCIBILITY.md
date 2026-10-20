# LittleLess AI — Independent Reproducibility Guide

**Author:** Putluru Sandeep ([@sunny-kavixo](https://github.com/sunny-kavixo))  
**Research Standard:** Deterministic, Bit-Exact, Zero-Omission Independent Replication  

---

## 1. System Specifications & Toolchain

All reported experimental results were generated on the following verified hardware and software environment:

- **Host Machine:** Apple MacBook Air (M4, 2024)
- **Processor:** Apple Silicon M4 (10-core CPU: 4 performance + 6 efficiency)
- **Unified Memory:** 16 GB Unified Memory (LPDDR5X, ~120 GB/s bandwidth)
- **Operating System:** macOS Sequoia 15.x (Darwin 24.x, arm64)
- **C++ Compiler:** Apple Clang version 17.x / 16.x (`clang++ -O3 -std=c++17 -shared -fPIC -lz`)
- **Python Runtime:** Python 3.12 / 3.14 (CPython arm64)
- **Key Dependencies:** `numpy >= 1.26`, `torch >= 2.2` (CPU evaluation), `zlib` (macOS system library)

---

## 2. Compilation Instructions

The native inference engine is implemented in standard C++17 with ARM NEON vector intrinsics and POSIX `pread()`. Compile the shared library from the repository root:

```bash
clang++ -O3 -std=c++17 -shared -fPIC -lz \
  src/native/native_packed_transformer.cpp \
  src/native/native_streaming_transformer.cpp \
  src/native/native_file_streaming_transformer.cpp \
  src/native/native_phase4_transformer.cpp \
  -o src/kernels/libnative_transformer.dylib
```

Verify that `src/kernels/libnative_transformer.dylib` was successfully generated:
```bash
file src/kernels/libnative_transformer.dylib
# Output: Mach-O 64-bit dynamically linked shared library arm64
```

---

## 3. Step-by-Step Reproduction Commands

### Step 1: Run the Complete Unit Test Suite (12 Tests)
Validates container parsing, CRC32 verification, Int8 KV-cache reduction, and deterministic token generation:

```bash
python3 -m unittest discover -s tests
# Expected Output: Ran 12 tests in ~0.08s -> OK
```

### Step 2: Run the Independent Audit Script
Directly inspects the compiled binary and verifies file sizes, native heap allocations, token identity, and multi-trial throughput:

```bash
python3 tests/audit_phase4_1_independent.py
```
**Expected Verification Output:**
```
[1] File Size Verification:
  S1 STRE Size:     523,756 bytes (Reported: 523,756 bytes) -> PASS
  S1 zlib Size:     443,924 bytes (Reported: 443,924 bytes) -> PASS
  C0 STRE Size:     572,338 bytes (Reported: 572,338 bytes) -> PASS

[2] Native C++ Memory Allocation Verification:
  Active Weight RAM: 193,664 bytes (Reported: 193,664 bytes) -> PASS
  Int8 KV-Cache RAM: 184,320 bytes (Reported: 184,320 bytes) -> PASS
  Total Model RAM:   386,304 bytes (Audited: 386,304 bytes)   -> PASS
  Container CRC32:   PASS

[3] S1 Token Correctness Verification:
  Generated: [105, 110, 32, 116, 104, 101, 32, 115, 101, 97]
  Status:    PASS (Bit-Exact)

[4] C0 Token Correctness Verification:
  Generated: [32, 44, 32, 97, 110, 100, 32, 116, 104, 101]
  Status:    PASS (Bit-Exact)

[5] Throughput Reproduction Benchmark:
  S1 STRE Throughput: > 2,000 tok/s -> PASS
  C0 STRE Throughput: > 2,100 tok/s -> PASS
```

### Step 3: Run Full Benchmark Suite (WikiText-2 Perplexity + All Candidates)
Executes all candidates across 35 independent trials and computes held-out test perplexity:

```bash
python3 benchmarks/run_phase4_1_breakthrough_benchmarks.py
```
Results will update `evidence/phase4_1_breakthrough_benchmarks.csv`.

### Step 4: Re-Run Mathematical Discovery Experiments (Phase 4.2)
Validate the mathematical falsification experiments for Hypotheses A, B, and C:

```bash
# Hypothesis A (SSTCD Truncated SVD)
python3 experiments/test_hypothesis_a_sstcd.py

# Hypothesis B (GF-PRS Finite Field Difference Entropy)
python3 experiments/test_hypothesis_b_gf_prs.py

# Hypothesis C (DDSQ Dyadic Power-of-Two KV Cache)
python3 experiments/test_hypothesis_c_ddsq.py
```

### Step 5: Regenerate Visualization Charts
```bash
python3 experiments/generate_research_charts.py
```
Generates SVG and high-resolution PNG charts in `docs/assets/`.
