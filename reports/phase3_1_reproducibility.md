# LittleLess AI — Phase 3.1 Reproducibility Guide

**Target Hardware:** Apple Silicon MacBook Air M4, 16.0 GB Unified RAM  
**Operating System:** macOS Darwin 24.x  
**Python Environment:** Python 3.14.5, `torch==2.13.0` (Apple MPS enabled), `numpy==2.4.6`  
**Workspace:** `/Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer/`

---

## 1. Directory Structure

```
LittleLess-AI-Phase3-Transformer/
├── benchmarks/
│   └── run_rigorous_benchmarks.py    # Master reproducible benchmark harness
├── checkpoints/
│   ├── transformer_ref_fp16.pt       # Trained 1.17M reference checkpoint
│   └── artifacts/
│       ├── AWG_DBQ_Hybrid_INT8.npz    # Primary champion compressed artifact (557,202 B)
│       ├── AWG_DBQ_Hybrid_INT8.json   # Deployment metadata
│       └── [other candidates]...
├── data/
│   └── wikitext-2/
│       ├── train.txt                 # Canonical WikiText-2 splits
│       ├── valid.txt
│       └── test.txt
├── evidence/
│   ├── phase3_verified_benchmarks.csv   # Verified benchmark ledger
│   └── phase3_1_experiment_results.csv  # Complete empirical measurements
├── reports/
│   ├── phase3_scientific_corrections.md # Audit and error corrections
│   ├── phase3_1_mathematical_hypotheses.md # AOCT & AWG-DBQ derivations
│   ├── phase3_1_pareto_frontier.md      # Multi-objective trade-off analysis
│   ├── phase3_1_reproducibility.md      # This guide
│   └── phase3_1_final_research_report.md# Master Phase 3.1 research deliverable
└── src/
    ├── awg_dbq.py                   # AWG-DBQ mathematical engine & packing
    ├── causal_transformer.py        # 1.17M Causal Transformer architecture
    ├── dataset.py                   # WikiText-2 non-overlapping sequence loader
    └── standalone_loader.py         # Standalone artifact loader & inference
```

---

## 2. Replication Commands

### Step 1: Verify Environment & Hardware
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__, '| MPS Available:', torch.backends.mps.is_available())"
```
*Expected Output:*
```
PyTorch: 2.13.0 | MPS Available: True
```

---

### Step 2: Verify Exact Parameter Count (1,172,928 parameters)
```bash
PYTHONPATH=. python3 -c "
from src.causal_transformer import CausalTransformerLM
model = CausalTransformerLM()
total = sum(p.numel() for p in model.parameters())
print(f'Total Causal Transformer Parameters: {total:,}')
assert total == 1172928, 'Parameter count mismatch!'
"
```
*Expected Output:*
```
Total Causal Transformer Parameters: 1,172,928
```

---

### Step 3: Run the Full Master Benchmark Suite
Executes all 10 candidates over the 192,000 held-out test tokens of WikiText-2, captures cold load time, prefill, decode latency, throughput, memory, and outputs `evidence/phase3_1_experiment_results.csv`:

```bash
PYTHONPATH=. python3 benchmarks/run_rigorous_benchmarks.py
```
*Key Benchmarks Verified:*
- `FP16_Reference`: PPL 6.370, Artifact 2,164,373 B, Decode ~1387 us/tok.
- `AWG_DBQ_Hybrid_INT8`: PPL 6.618 (+3.89% gap), Artifact 557,369 B (< 600 KB), Decode ~1358 us/tok (736.5 tok/s). All gates: **PASS**.

---

### Step 4: Verify Standalone Deserialization & Text Generation
Test loading the model strictly from the `.npz` artifact without reference weights:

```bash
PYTHONPATH=. python3 -c "
import torch
from src.standalone_loader import load_standalone_transformer

model, meta = load_standalone_transformer('checkpoints/artifacts/AWG_DBQ_Hybrid_INT8.npz')
prompt = torch.tensor([[ord(c) for c in 'The European lobster is ']], dtype=torch.long, device='mps')
out = model.generate(prompt, max_new_tokens=48)
text = bytes(out[0].cpu().tolist()).decode('latin1', errors='replace')
print('Generated Sample:', repr(text))
"
```
*Expected Output:* Coherent autoregressive text generation with no runtime dequantization crashes.
