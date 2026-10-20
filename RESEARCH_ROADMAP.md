# LittleLess AI research roadmap

**Status: ongoing.** Phase numbers track experimental programs, not completion percentages.

| Priority | Workstream | Acceptance criterion | Status |
|---|---|---|---|
| P0 | Recover and audit Phase 4.0 Python/native crash | Identify affected process, exit reason, data integrity, and safe rerun path | Open |
| P0 | Independent Phase 3.9 reproduction | Rebuild native library, run safety tests, inspect file I/O, compare output and memory | Open |
| P0 | Public reproducibility package | Review and publish permitted source, test scripts, dataset instructions, hashes, and raw evidence | Open |
| P1 | End-to-end memory accounting | Report native allocations, KV cache, peak RSS, and OS file-cache effects separately | Open |
| P1 | Compressed file-backed streaming | Test chunked entropy coding with bounded decompression and direct seeking | Planned |
| P1 | KV-cache compression | Measure accuracy, context scaling, memory, and latency on controlled tasks | Planned |
| P1 | Mathematical invention track | Three falsifiable distinct hypotheses with prior-art checks and ablations | Planned |
| P2 | Larger model transfer | Reproduce benefits on progressively larger, licensed models | Planned |
| P2 | External baseline comparison | Matched LittleBit/LittleBit-2 and conventional baselines; no unsupported claims | Planned |

## Experiment workflow

Hypothesis → prior-art search → implementation → correctness tests → benchmark → independent criticism → revise or reject.

For each experiment preserve commit SHA, dataset, split, seed, hardware, toolchain, raw CSV, plots, negative results, and measurement uncertainty.

## Publication policy

Research results are published after security/licensing review and reproducibility checks. Do not push private local files, credentials, unlicensed weights, or unreviewed binary artifacts. The research workspace and Git history must be preserved.
