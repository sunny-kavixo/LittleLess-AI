# LittleLess AI
### Independent research in AI compression, compact model representations, and low-memory inference

**Researcher:** Putluru Sandeep ([@sunny-kavixo](https://github.com/sunny-kavixo))  
**Status:** Active, experimental research — not a production-ready model or a proven replacement for existing compression systems.

## Why LittleLess exists

Modern AI models require substantial storage, memory bandwidth, RAM, and compute. LittleLess investigates whether new mathematical representations, binary storage formats, and execution methods can reduce those costs **without unacceptable losses in model quality**.

The long-term goal is to develop genuinely original, measurable compression and inference mechanisms that could eventually scale from small experimental transformers to larger language models. Established methods, including LittleBit/LittleBit-2 and conventional low-bit quantization, are research baselines to study and compare against — **not achievements already surpassed**.

## Research questions

1. Can learned or structured low-bit representations retain quality at lower storage cost?
2. Can entropy-coded weights be decoded incrementally without keeping the entire packed model in RAM?
3. Can file-backed execution lower active model-weight allocations without unacceptable I/O or latency overhead?
4. Can attention/KV-cache memory scale more efficiently with context length?
5. Which changes improve the **joint** trade-off among perplexity, artifact size, peak process memory, throughput, and compute?
6. Which proposed mechanisms are genuinely new, rather than combinations of known quantization, sparsity, entropy coding, and streaming techniques?

## Research progression

| Stage | Focus | Current evidence/status |
|---|---|---|
| Phase 3.1 | AWG-DBQ hybrid low-bit representation | Five acceptance gates reported as passed; independent replication should be documented |
| Phases 3.3–3.5 | ADBO, quality optimization, Radix-3 packing | Experimental implementation and benchmark reports |
| Phase 3.6 | S-ADBO sparsity | Reported 70.94% zero-trit coefficients, but worse perplexity; quality/storage trade-off |
| Phase 3.7 | Entropy-coded `.lpack` representation | Reported 393,071-byte serialized artifact; disk size is not active RAM |
| Phase 3.8 | Native JIT weight streaming | Independent audit identified retained Python weight buffers, incomplete memory accounting, and pointer-lifetime risk |
| Phase 3.9 | Native file-backed `.llstream` execution | Local independent-audit script reported PASS, including FD invalidation, CRC, lifecycle, and output checks; external reproduction pending |
| Phase 4.0 | Chunk-streamed entropy coding and dynamic Int8 KV cache | Local benchmark completed; 11 reported rows and crash diagnosis documented; public reproduction pending |

These are **research-log-reported** results, not independently certified claims. Different configurations and evaluation conditions must not be compared as if they were identical.

## Selected reported Phase 3.9 measurements

| Metric | Reported result | Interpretation |
|---|---:|---|
| Native active model-weight allocation, Radix-3 layer streaming | 160,704 bytes (~156.9 KiB) | **Not** total process memory |
| `.llstream` Radix-3 container | 572,140 bytes | Larger than the earlier entropy-coded `.lpack` artifact |
| Weight reads per generated token | 50 `pread()` calls | File-backed access telemetry |
| Bytes streamed per token, Radix-3 | 443,136 | Storage I/O cost |
| Radix-3 S1 decode | ~3,233 tokens/s | Small experimental transformer; not representative of a large LLM |
| C0 held-out perplexity | 6.4954 | Reported evaluation; model/dataset details must accompany reproduction |
| Python/PyTorch process RSS | ~397 MB | Native allocation reduction does **not** imply a sub-200-KB application |

Source: local Phase 3.9 research report, pending publication of its complete reproducibility artifacts. No broad large-model or real-world speedup is claimed.

## Latest Phase 4.0 findings (local research report)

- **Dynamic Int8 KV cache:** 655,360 → 184,320 bytes at context 128 (**71.9% lower KV allocation**).
- **Native model heap:** 824,384 → 353,344 bytes for the raw-stream Int8 variant (**57.1% lower**, not process RSS).
- **Sparse S1 chunk container:** 572,140 → 443,924 bytes (**22.4% smaller**) at the cost of decode speed falling from ~3,243 to ~961 tokens/sec.
- **Dense C0 chunk compression:** marginal disk benefit with a substantial speed penalty; this hypothesis was not useful for the dense candidate.
- **Crash clarification:** a temporary Python `ctypes` diagnostic reportedly used an incorrect pointer signature; benchmark jobs themselves were reported to exit successfully. See [CRASH_ANALYSIS.md](CRASH_ANALYSIS.md).

These figures are **reported from local experiments**, not independently reproduced from this public repository. Read [PHASE4_RESULTS.md](PHASE4_RESULTS.md) and [RESULTS.md](RESULTS.md) before quoting them.

## What has not been solved

- A new compression theory with demonstrated superiority over established baselines has **not** been proven.
- Phase 3.8's original low-total-memory interpretation failed independent scrutiny.
- Phase 3.9 trades compact entropy-coded disk size for seekable file-backed execution; actual peak RSS, page-cache effects, and cross-process reproducibility need careful evaluation.
- Sparsity improvements can worsen perplexity.
- KV-cache cost grows with sequence length; it may dominate active model-weight allocation.
- Small-model tokens/second cannot be extrapolated to large language models.
- A separate Python/native crash was traced locally to an ad-hoc `ctypes` binding mistake; typed FFI wrappers and broader sanitizer tests remain pending.

See [RESULTS.md](RESULTS.md), [PHASE4_RESULTS.md](PHASE4_RESULTS.md), [LIMITATIONS.md](LIMITATIONS.md), [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md), and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Research standards

Every meaningful result should identify the Git commit, dataset/split, model configuration, seed, hardware, exact command, raw measurement, and comparison baseline. Failed hypotheses and regressions are recorded rather than hidden. Claims are classified as **verified**, **reported**, **partially verified**, **falsified**, or **planned**.

## Source and reproduction status

The public repository is being established as the research record. Local experimental source, benchmarks, reports, and test artifacts must be reviewed for correctness, licensing, secrets, and reproducibility **before** publication. A README entry alone is not evidence that an experiment can be reproduced from this repository.

## Vision

**Less storage. Less memory. Less compute. Preserved capability — proven through measurements, not slogans.**

This is independent, ongoing research by Putluru Sandeep. No affiliation with the authors of external baselines is implied.
