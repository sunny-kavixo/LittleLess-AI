# Limitations, negative results, and open problems

This document records scientific limitations rather than treating every experiment as a success.

## Quality versus compression
- Phase 3.6 S-ADBO reportedly reached 70.94% zero-trit coefficients, while perplexity worsened (reported S1 ~7.1464 versus C0 ~6.4954 in later comparable small-transformer tests). Sparse does not automatically mean better.
- Results from different datasets, checkpoints, seeds, or precision modes must not be compared without controlled conditions.

## Storage versus active memory
- Phase 3.7's reported 393,071-byte entropy-coded `.lpack` artifact is a disk-size measurement.
- Phase 3.9's reported 160,704-byte native active-weight allocation is a *subset* of memory, not whole-process RSS.
- Phase 3.9 Radix-3 `.llstream` is reported at 572,140 bytes, reflecting a trade-off for random-access streaming.
- Python/PyTorch evaluation was reported at approximately 397 MB RSS. Page cache, shared libraries, KV cache, native allocations, and peak RSS require separate accounting.

## Safety and reproducibility
- Phase 3.8 independent audit found Python pinned-weight retention and pointer ownership risks. It should not be presented as verified true disk streaming.
- Phase 3.9 reports file-descriptor-based native reads, corruption checks, and create/free tests; independent reruns and native safety tooling remain desirable.
- Subsequent local crash investigation attributed the macOS Python crash to an ad-hoc `ctypes` call missing the `c_void_p` signature for `free_phase4_model`. Benchmark tasks reportedly exited successfully. This does not rule out other native defects; see [CRASH_ANALYSIS.md](CRASH_ANALYSIS.md).

## Phase 4.0 measured trade-offs (locally reported)
- Dynamic Int8 KV cache reduced native KV allocation by ~71.9% at T=128, but longer-context scaling and actual whole-process peak RSS are not yet established.
- Sparse S1 chunk compression reduced container size by ~22.4% but reduced throughput from ~3,243 to ~961 tokens/s.
- Dense C0 chunk compression offered only ~0.50% total container savings by the table's byte counts, despite a large speed penalty; the report separately states ~0.69%, so this discrepancy needs reconciliation.
- Perplexity and tokens were reported close/matching for tested configurations; this does not imply no degradation across arbitrary tasks or models.

## Scale and novelty
- Current reported results involve a small experimental transformer. No comparable large-language-model superiority has been established.
- Radix-3 packing, rANS, quantization, sparsity, and `pread()` are established techniques. A new combination is not automatically a novel mathematical invention.
- Any LittleBit/LittleBit-2 superiority claim requires matched models, datasets, metrics, and reproducible baselines.

## Evidence labels
**Verified** = independently reproduced with inspectable evidence. **Reported** = research agent or log output awaiting independent reproduction. **Partial** = some criteria met, others unresolved. **Falsified** = contradicted by subsequent audit. **Planned** = not yet demonstrated.
