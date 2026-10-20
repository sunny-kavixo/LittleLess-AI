# Phase 4.0 — Experimental results and interpretation

**Evidence classification:** Local agent-reported benchmark and audit. This GitHub documentation has not independently executed the native code. Raw CSV and source are not yet published here.

**Local reported commits:** Phase 3.9 `d2eda4b` (`phase-3.9-file-backed-streaming`); Phase 4.0 `eea4044` (`phase-4.0-research`). These are **not commits in this public repository**.

**Environment reported:** MacBook Air M4, 16 GB unified memory; Apple Clang 21, C++17, PyTorch 2.13, NumPy 2.4.6; WikiText-2 byte-level tokenizer, vocabulary 256, 192,000 held-out evaluation tokens. Experimental model context T=128.

## Goal

Jointly reduce serialized model size, native runtime heap allocations, KV-cache memory, and inference cost while retaining measured output quality. This is a small-transformer study, **not** a validated large-LLM compression result.

## Experimental mechanisms

1. **Chunk-streamed entropy coding (`.llchunk`):** Independently zlib/DEFLATE-compressed ternary weight chunks indexed for bounded `pread()` and decompression. Standard techniques applied to the experimental native streaming engine.
2. **Dynamic Int8 KV cache:** Per-head, per-timestep scale for key/value vectors, with quantized Int8 values and dequantization during attention. An integration experiment, not a proven novel quantization algorithm.
3. **Low-rank residual correction hypothesis:** `W ≈ s ⊙ T + U Vᵀ`, with ternary `T` and rank-2 factors. **Proposed, not implemented or experimentally validated** in Phase 4.0.

## Benchmark matrix — locally reported

All numbers are from the provided Phase 4.0 research report; decode throughput is specific to the test model and setup.

| Candidate | Container bytes | Native active weight B | KV cache B | Total native model heap B | Test PPL | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| C0, raw stream, FP32 KV | 572,140 | 160,704 | 655,360 | 824,384 | 6.4954 | 3,251.1 |
| C0, chunk stream, FP32 KV | 569,290 | 177,184 | 655,360 | 840,864 | 6.4954 | 866.5 |
| C0, raw stream, Int8 KV | 572,140 | 160,704 | 184,320 | 353,344 | 6.4952 | 3,235.1 |
| C0, chunk stream, Int8 KV | 569,290 | 177,184 | 184,320 | 369,824 | 6.4952 | 871.6 |
| S1, raw stream, FP32 KV | 572,140 | 160,704 | 655,360 | 824,384 | 7.1462 | 3,242.9 |
| S1, chunk stream, FP32 KV | 443,924 | 177,184 | 655,360 | 840,864 | 7.1462 | 961.4 |
| S1, raw stream, Int8 KV | 572,140 | 160,704 | 184,320 | 353,344 | 7.1463 | 3,246.6 |
| S1, chunk stream, Int8 KV | 443,924 | 177,184 | 184,320 | 369,824 | 7.1463 | 958.9 |

Three additional S1 chunk+Int8 trials (seeds 42, 1337, 2026) reported 960.8, 945.6, and 963.7 tok/s respectively. The source report calls these seeds, but further review should verify exactly which randomness is controlled.

### What improved

- **KV cache:** 655,360 → 184,320 B (71.875% reduction) at T=128 in this test.
- **Native model heap, raw-stream Int8 variant:** 824,384 → 353,344 B (57.14% reduction). This is **not** total process RSS.
- **Sparse S1 serialized container:** 572,140 → 443,924 B (22.41% reduction), using chunk compression.
- **Sparse S1 streamed bytes per token:** 443,136 → 314,720 B (28.98% reduction).
- **Reported perplexity:** Int8 KV changed S1 from 7.1462 to 7.1463, a small measured difference, not a universal guarantee of no quality loss.

### What regressed

- Sparse S1 chunk-stream throughput: ~3,242.9 → ~961.4 tok/s with FP32 KV (~70% lower).
- Dense C0 chunk coding: only ~0.50% smaller total container (572,140 → 569,290 B), while throughput fell ~3.75×. The source report separately describes ~0.69% savings; the listed artifact bytes imply ~0.50%. **The discrepancy should be reconciled against raw CSV and definitions.**
- Chunk decoding increases native active weight/scratch allocation from 160,704 to 177,184 B.
- S1 perplexity remains materially worse than C0; sparsity/size improvements do not imply improved model quality.

## Phase 3.9 audit context

Local audit reported: file-descriptor invalidation caused read failure; 200 create/free cycles with no FD drift; CRC32 corruption detection; reported exact generated-token agreement; 160,704 B native active weight allocation. These are valuable local tests, but independent external replication remains outstanding.

## Python crash clarification

A macOS `EXC_BAD_ACCESS (SIGSEGV)` occurred in an **ad-hoc Python ctypes diagnostic**, not the completed Phase 4.0 benchmark suite according to the local investigation. The diagnostic omitted `lib.free_phase4_model.argtypes = [ctypes.c_void_p]`, allowing a 64-bit native pointer to be passed with an incorrect default integer conversion before deletion. The crash report identified `NativePhase4Model::~NativePhase4Model()` / `free_phase4_model`; the diagnostic was rerun with an explicit pointer signature and reportedly completed successfully.

The benchmark tasks were reported to have exited 0, with an intact 11-row CSV. This supports a **specific FFI binding misuse** explanation; it does not prove that all native C++ code is free of memory-safety bugs. A typed binding wrapper and sanitizer checks are sensible follow-ups.

## Evidence needed for public reproduction

- Review and publish `src/native/native_phase4_transformer.cpp/.h`, `src/export_phase4_chunk_container.py`, `tests/test_phase4_research.py`, `tests/audit_phase3_9_independent.py`, `benchmarks/run_phase4_research_benchmarks.py`, and `evidence/phase4_research_benchmarks.csv` after licensing/security checks.
- Add exact build commands, dependency versions, checkpoint provenance and hashes, hardware, test invocation, and independent output logs.
- Measure process peak RSS and page-cache effects separately from native model heap.
- Verify the KV implementation is a true incremental cache at the stated context length; test longer contexts and real tasks.
- Compare against existing KV quantization and streaming implementations before making novelty claims.

**Conclusion:** A promising measured engineering trade-off: substantial local KV/native-heap reductions, modest sparse container savings, and a significant decompression speed cost. Larger-model applicability and fundamental mathematical novelty remain open.
