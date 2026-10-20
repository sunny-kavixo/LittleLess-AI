# Reproducibility and Evidence Guide

This repository contains the complete experimental code, native C++ kernels, benchmark harness, unit tests, and empirical evidence CSVs for LittleLess AI.

## 1. Verified Experimental Environment

- **Architecture:** Apple Silicon (macOS, arm64, Apple M4, 16 GB unified memory).
- **Toolchain:** Apple Clang (C++17), `-O3 -fPIC -shared -std=c++17`.
- **Python Runtime:** Python 3.10+ (tested on Python 3.10 / 3.14).
- **Core Dependencies:** `numpy`, `torch` (for PyTorch validation baseline), `matplotlib` (for chart generation).
- **Evaluation Dataset:** WikiText-2 raw tokens (vocabulary: 256 byte-level tokens, 192,000 held-out evaluation tokens, sequence context $T=128$).
- **License:** All source code and documentation are released under the Apache-2.0 License.

---

## 2. Native Compilation Instructions

Compile the unified native inference kernel before running tests or benchmarks:

```bash
# Build the native C++ transformer dynamic library
clang++ -O3 -fPIC -shared -std=c++17 \
  -I src/native \
  src/native/native_packed_transformer.cpp \
  src/native/native_streaming_transformer.cpp \
  src/native/native_file_streaming_transformer.cpp \
  src/native/native_phase4_transformer.cpp \
  -lz \
  -o src/kernels/libnative_transformer.dylib
```

---

## 3. Running Unit Tests and Audits

Execute the automated test suite and independent verification gates:

```bash
# 1. Run all unit tests (12/12 test cases)
python3 -m unittest discover -s tests

# 2. Run Phase 3.9 Independent Verification Gate
python3 tests/audit_phase3_9_independent.py

# 3. Run Phase 4.1 Independent Breakthrough Verification Gate
python3 tests/audit_phase4_1_independent.py

# 4. Run Phase 4.2 Hypothesis Testing Scripts
python3 experiments/test_hypothesis_a_sstcd.py
python3 experiments/test_hypothesis_b_gf_prs.py
python3 experiments/test_hypothesis_c_ddsq.py
```

---

## 4. Benchmark Reproduction

Run the audited benchmarking suites:

```bash
# Profile Phase 4.0 execution breakdown (pread vs decompress vs gemv)
python3 benchmarks/profile_phase4_breakdown.py

# Run Phase 4.1 breakthrough benchmark (evaluates raw, chunked, and STRE modes)
python3 benchmarks/run_phase4_1_breakthrough_benchmarks.py

# Re-generate research charts and visual assets
python3 experiments/generate_research_charts.py
```

---

## 5. Artifact and Checkpoint Provenance

Model checkpoints and raw binary datasets are excluded from Git to prevent repository bloat (`checkpoints/` and `data/` in `.gitignore`).

To export the Phase 4.1 STRE container from an existing checkpoint:
```bash
# Export STRE container
python3 src/export_phase4_1_stre_container.py \
  --checkpoint checkpoints/model_phase3.npz \
  --output checkpoints/model_phase4_stre.llstre
```

---

## 6. Minimum Evidence for Validated Results

Every claim in this repository adheres to the following scientific standards:
1. **Source Tracking:** Exact Git commit and inspectable source code.
2. **Explicit Provenance:** Model architecture, parameter count, dataset split, sequence length, and seed.
3. **Execution Separation:** Complete separation between native C++ heap allocations (measured via native tracking counters) and whole-process Python RSS (measured via `getrusage`).
4. **Independent Reruns:** Deterministic bit-exact token verification between PyTorch reference and native C++ kernels across multiple independent trials.
5. **Rigorous Classification:** Explicit distinction between **Verified** results (independently reproduced), **Preliminary** results, and **Falsified** hypotheses.
