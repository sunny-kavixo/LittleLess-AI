# LittleLess AI — Phase 3 Reproducibility Guide

## 1. Environment & Hardware
- **Processor**: Apple Silicon M4 (10-core CPU, 16-core NPU)
- **Memory**: 16.00 GB LPDDR5X Unified RAM
- **OS**: macOS Darwin 24.x
- **Python**: 3.14.5
- **Libraries**: `torch==2.13.0` (MPS enabled), `numpy==2.4.6`

---

## 2. Directory Layout
```bash
cd /Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer
```

---

## 3. End-to-End Execution Commands

### Step 1: Verify Previous A-DBQ Evidence
```bash
python3 src/verify_step1_adbq.py
```
*Expected Output*:
- Reproduces FP16 PPL 10.930, Full Ternary PPL 13.741, Global A-DBQ PPL 11.250 (16,606 bytes), PC A-DBQ PPL 11.224 (18,453 bytes).
- Verifies exact reconciliation between 43,967 bytes (compressed zip) and 49,109 bytes (uncompressed zip).

---

### Step 2: Run Full 1.17M Transformer Benchmark Suite on WikiText-2
```bash
PYTHONPATH=. python3 src/train_and_evaluate_transformer.py
```
*Expected Output*:
- Evaluates the 1.17M Causal Transformer on WikiText-2 test set (192,000 tokens).
- Reference FP16 Perplexity: **6.370** (2,169,343 bytes).
- Tests all 13 candidate architectures, sensitivity ablations, negative controls, and generated text samples.
- Outputs complete benchmark metrics to `evidence/transformer_compression_benchmark.csv`.
