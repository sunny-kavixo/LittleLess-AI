# Reproducibility and evidence policy

This public repository currently documents the research program. The local experimental code and datasets are not yet guaranteed to be available here.

## Reported local research environment

- Hardware: Apple Silicon MacBook Air M4, 16 GB unified memory.
- Research workspace (local only): `LittleLess-AI-Phase3-Transformer`.
- Reported Phase 3.9 branch: `phase-3.9-file-backed-streaming`.
- Reported Phase 3.9 commit: `d2eda4b` (not yet confirmed as a commit in this public repository).
- Native implementation: C++17, Clang, POSIX `pread()`, Python benchmark harness.
- Model: small experimental transformer; published claims must specify full architecture and checkpoint provenance.

## Reported Phase 3.9 local test commands

These commands require source files, dependencies, and model artifacts that may **not yet exist in this public repository**.

```bash
python3 tests/test_file_streaming_safety.py
python3 tests/test_radix3_exhaustive.py
python3 benchmarks/run_phase3_9_file_streaming_benchmarks.py
```

The original local benchmark environment used a custom Python package path. Do not hardcode private absolute paths into a public reproducibility guide.

## Minimum evidence for a validated result

1. Exact Git commit and source archive hash.
2. Model architecture, parameter count, source checkpoint and license.
3. Dataset, split, preprocessing, sequence length, and random seeds.
4. Hardware, OS, Python, compiler, dependencies, and native build flags.
5. Exact commands and raw CSV/log output.
6. Multiple trials and timing methodology.
7. Serialized artifact bytes; native active weights; scratchpad; KV cache; peak process RSS; page-cache considerations.
8. Perplexity and output-agreement methodology.
9. Failure cases, crashes, and independently observed regressions.
10. An explicit distinction between **reported** and **independently reproduced** results.

## Publication checklist

Before importing local research: inspect `git status`, review staged files and commit history, scan for credentials, check dataset/model licenses, avoid giant binaries, and preserve historical results. Never replace missing evidence with invented commands or numbers.
