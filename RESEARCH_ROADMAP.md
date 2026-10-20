# LittleLess AI research roadmap

**Status: ongoing.** Phase numbers track experimental programs, not completion percentages.

| Priority | Workstream | Acceptance criterion | Status |
|---|---|---|---|
| P0 | Phase 4.0 crash root-cause follow-up | Local diagnosis points to ad-hoc `ctypes` pointer truncation; add typed FFI wrapper and sanitizer checks | Diagnosed locally; hardening open |
| P0 | Independent Phase 3.9 reproduction | Local audit reported PASS for file I/O, CRC, lifecycle and output; publish scripts and repeat externally | Local PASS; public reproduction open |
| P0 | Public reproducibility package | Review and publish permitted source, test scripts, dataset instructions, hashes, and raw evidence | Open |
| P1 | End-to-end memory accounting | Report native allocations, KV cache, peak RSS, and OS file-cache effects separately | Open |
| P1 | Compressed file-backed streaming | Phase 4.0 chunked DEFLATE implemented locally; optimize decode overhead and compare against alternatives | Experimental result; speed optimization open |
| P1 | KV-cache compression | Phase 4.0 dynamic Int8 KV measured at T=128; validate incremental semantics, longer contexts and real tasks | Experimental result; scaling open |
| P1 | Mathematical invention track | Three proposals recorded; low-rank residual hypothesis not yet implemented; novelty requires prior-art and ablations | In progress |
| P2 | Larger model transfer | Reproduce benefits on progressively larger, licensed models | Planned |
| P2 | External baseline comparison | Matched LittleBit/LittleBit-2 and conventional baselines; no unsupported claims | Planned |

## Experiment workflow

Hypothesis → prior-art search → implementation → correctness tests → benchmark → independent criticism → revise or reject.

For each experiment preserve commit SHA, dataset, split, seed, hardware, toolchain, raw CSV, plots, negative results, and measurement uncertainty.

## Publication policy

Research results are published after security/licensing review and reproducibility checks. Do not push private local files, credentials, unlicensed weights, or unreviewed binary artifacts. The research workspace and Git history must be preserved.
