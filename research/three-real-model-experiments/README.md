# LittleLess AI — Three Real-Model Experiments (Evidence Preview)

**Researcher:** Putluru Sandeep · **Project:** https://github.com/sunny-kavixo/LittleLess-AI  
**Release status:** Evidence preview, NOT independently reproducible end-to-end from this package.  
**Scope:** Only TinyMoE pretrained inference, SmolLM2-135M quantization, and 14B disk-offloaded inference. No synthetic benchmarks or unrelated project claims.

This package distinguishes records **actually available and integrity-checked** from older reported measurements whose original evidence is currently **not included**. No external replication, world-first invention, new compression algorithm, preserved LLM quality, or production-ready deployment is claimed.

## Result ledger

| Experiment | Reported outcome | Source/evidence status | Meaning and constraints |
|---|---|---|---|
| TinyMoE v0.2 disk-backed experts | Full-load peak process RSS **1,453.15 MiB**; streamed **1,254.53 MiB** (**13.67%** reduction, **198.62 MiB** saved). **6.94 vs 3.41 tok/s**. Four generated token IDs and recorded logits matched. | **HISTORICAL REPORT ONLY.** Original v0.2 code, weights, logs, and evidence archive not included here; independent replay impossible from this package. | A narrow CPU/pretrained-model proof of concept; streaming ~2.04× slower; process RSS != full-system usage. |
| SmolLM2-135M-Instruct dynamic/selected INT8 (Gate 4) | FP32 `state_dict` **513.236 MiB**; LAST8/LSGQ8 **452.504 MiB** (**11.83%** smaller). Gate 4 A/B/C/D next-token scores FP32 **21/100**, LAST8 **30/100**, LSGQ8 **25/100**. | **HISTORICAL EXPERIMENT SUMMARY**; original final run result files and pretrained weights not included. A separate dataset and runner are not a replay of frozen results. | Storage reduction is plausible, but quality preservation not proven; scores near random choice; quantization is established technique. Gate 4 measured higher process peak RSS for quantized conversion paths. |
| 14B disk-offloaded Transformers/Accelerate | **Two generated tokens**, measured generation **274.514 s**, **0.007286 tok/s**; sampled process RSS **8212.27 MiB** (~8.02 GiB). | **ARCHIVED RAW EVIDENCE INCLUDED**, checksums validated. Model shards, runner source, tokenizer and pinned model revision absent, so full replay cannot be claimed. | Feasibility of a two-step offloaded forward, not practical inference; 4.5 GiB configuration budget is NOT observed RSS; not proof of a memory-lowering compression algorithm. |

### Why this is not a benchmark leaderboard

Different models, token counts, inference routes and measurement procedures make between-row ranking invalid. For SmolLM2, the score depended on choice ordering and scoring method. TinyMoE evidence reports only one short prompt. The 14B offload route was extremely slow and lacked full-load parity. The important positive outcome is implementation and documented trade-offs, not state-of-the-art superiority.

## What can be verified offline *now*

1. `python scripts/verify_release.py` validates the included original 14B ZIP against its recorded internal SHA-256 manifest, JSON metrics, logits vector shape and finite entries. Does not load a language model or require network access.
2. The archived 14B evidence has two logits arrays, each with a 152,064-element vocabulary vector, plus per-token ID and latency records.
3. Source archives not included here are **not** silently reconstructed. When original TinyMoE v0.2 or SmolLM2 benchmark evidence becomes available, store it separately and independently audit it **before** changing their status to verified.

## How to publish responsibly

- GitHub: publish as `research/three-real-model-experiments/`, alongside original archives only when available, with this provenance ledger and verification script. Link to the primary LittleLess repository.
- Hugging Face: an optional research report / dataset repository for these measurements, **not a pretrained model release**. Do not create a model repository implying a downloadable LittleLess model exists.
- Reproduction tiers: `archive-integrity VERIFIED`, `metrics REPORTED`, `end-to-end reproduction NOT VERIFIED`, `independent replication NOT VERIFIED`.

## Reproduction and dependencies

For **archival integrity only**: Python 3.9+, NumPy. `python scripts/verify_release.py`. No model downloads.

Full-model reproduction cannot be supplied honestly until model revision, complete runner, dependency lock, model weights acquisition instructions and missing archives are recovered. Never conflate a ZIP SHA256 check with independent experimental replication.

## Research boundaries

- All measurements are from reported CPU tests; no external replication.
- Peak process RSS does not include OS page cache and is subject to sampling resolution.
- No claim of novel algorithmic invention, 6 GiB 14B inference, quality-parity on held-out evaluations, or lower RAM alongside faster generation.
- The package is a **publication-ready evidence preview with disclosures**, **not an independently runnable model inference release**.
