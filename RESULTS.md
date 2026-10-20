# Research results index

**Source status:** These tables transcribe the owner's local agent-generated research reports. Source code, benchmark CSVs, checkpoint hashes, and independent reruns have not yet been published in this repository. Treat results as **reported**, not externally verified.

| Phase | Reported milestone | Key limitation |
|---|---|---|
| 3.1 | AWG-DBQ hybrid: five acceptance gates passed | Full reproducibility package pending |
| 3.3–3.5 | ADBO, Radix-3 low-bit representation | Novelty and baseline comparison incomplete |
| 3.6 | S-ADBO: ~70.94% zero trits | Worse perplexity |
| 3.7 | rANS `.lpack`: 393,071 B artifact | Disk compression does not equal RAM reduction |
| 3.8 | Native JIT streaming | Independent audit found Python weight retention and pointer ownership issues |
| 3.9 | File-backed `pread()` streaming: 160,704 B native active weights | Full RSS and OS page-cache effects separate |
| 4.0 | Dynamic Int8 KV: 71.875% less KV allocation; chunk compression | Chunk decompression slows inference substantially |

**Phase 4.0:** See [PHASE4_RESULTS.md](PHASE4_RESULTS.md) for full reported matrix, experimental conditions, negative results, and crash clarification.

## Interpretation

Research quality requires comparing matching model, checkpoint, evaluation split, context length, hardware, and execution mode. The smaller S1 artifact does not compensate for its worse perplexity unless an explicit quality/size trade-off is accepted. Native heap allocations are not whole-process RSS.

## Current public evidence gap

The local report references `evidence/phase4_research_benchmarks.csv` and native implementation files, but they are not yet hosted in this repository. A local commit hash is not a GitHub source link until that commit is published. Do not label these results externally reproduced.
