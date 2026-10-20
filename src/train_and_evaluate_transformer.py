"""
Step 2, 3, 4, 5: Complete Causal Transformer Training, Compression Suite & Performance Benchmarks
Evaluates 11 candidate mechanisms on WikiText-2 across validation and held-out test splits.
Target: 1.17M Causal Transformer, MacBook Air M4 (16GB RAM).
"""

import csv, hashlib, io, json, math, os, resource, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.causal_transformer import CausalTransformerLM, apply_weight_quant
from src.dataset import get_dataloaders

ROOT = Path('/Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer')
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

def load_or_train_reference(epochs=3, lr=1e-3, seed=42):
    ckpt_path = ROOT / 'checkpoints/transformer_ref_fp16.pt'
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=ROOT / 'data/wikitext-2',
        block_size=128,
        batch_size=32,
        train_samples=10000,
        eval_samples=1500
    )
    
    model = CausalTransformerLM().to(DEVICE)
    if ckpt_path.exists():
        print(f"Loading existing trained reference model from {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        return model, val_loader, test_loader
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"--- Training 1.17M Causal Transformer on WikiText-2 (Device: {DEVICE}, Epochs: {epochs}) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, l = model(x, y)
                val_loss += l.item()
        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.3f}")
        
    train_time = time.perf_counter() - t0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Reference model trained in {train_time:.1f} s and saved to {ckpt_path}\n")
    return model, val_loader, test_loader

def evaluate_perplexity(model, dataloader, quant_modes={}):
    model.eval()
    model.set_quant_modes(quant_modes)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    mean_ce = total_loss / total_tokens
    ppl = math.exp(mean_ce)
    return mean_ce, ppl

def pack_matrix_to_bytes(arr, mode):
    payload = {}
    if arr.ndim < 2 or mode == 'fp16':
        payload['val'] = arr.astype(np.float16)
        return payload

    if mode == 'ternary_global':
        m = np.abs(arr) > 0.7 * np.abs(arr).mean()
        s = float(np.abs(arr[m]).mean()) if m.any() else 1.0
        codes = np.where(m, np.sign(arr), 0).astype(np.int8).ravel() + 1
        bits = np.stack([(codes >> 1) & 1, codes & 1], 1).ravel()
        payload['bits'] = np.packbits(bits, bitorder='little')
        payload['scale'] = np.array([s], dtype=np.float16)
    elif mode == 'ternary_pc':
        thresh = 0.7 * np.abs(arr).mean(axis=-1, keepdims=True)
        m = np.abs(arr) > thresh
        s = (np.abs(arr) * m).sum(axis=-1, keepdims=True) / np.maximum(1, m.sum(axis=-1, keepdims=True))
        codes = np.where(m, np.sign(arr), 0).astype(np.int8).ravel() + 1
        bits = np.stack([(codes >> 1) & 1, codes & 1], 1).ravel()
        payload['bits'] = np.packbits(bits, bitorder='little')
        payload['scale'] = s.astype(np.float16)
    elif mode in ('dbq_global', 'dbq_pc'):
        is_pc = (mode == 'dbq_pc')
        axis = -1 if is_pc else None
        # Plane 1
        t1 = 0.7 * np.abs(arr).mean(axis=axis, keepdims=True) if is_pc else 0.7 * np.abs(arr).mean()
        m1 = np.abs(arr) > t1
        s1 = (np.abs(arr) * m1).sum(axis=-1, keepdims=True) / np.maximum(1, m1.sum(axis=-1, keepdims=True)) if is_pc else float(np.abs(arr[m1]).mean())
        q1 = s1 * np.sign(arr) * m1
        c1 = np.where(m1, np.sign(arr), 0).astype(np.int8).ravel() + 1
        bits1 = np.stack([(c1 >> 1) & 1, c1 & 1], 1).ravel()
        
        # Plane 2
        res = arr - q1
        t2 = 0.7 * np.abs(res).mean(axis=axis, keepdims=True) if is_pc else 0.7 * np.abs(res).mean()
        m2 = np.abs(res) > t2
        s2 = (np.abs(res) * m2).sum(axis=-1, keepdims=True) / np.maximum(1, m2.sum(axis=-1, keepdims=True)) if is_pc else float(np.abs(res[m2]).mean())
        c2 = np.where(m2, np.sign(res), 0).astype(np.int8).ravel() + 1
        bits2 = np.stack([(c2 >> 1) & 1, c2 & 1], 1).ravel()
        
        payload['p1'] = np.packbits(bits1, bitorder='little')
        payload['s1'] = s1.astype(np.float16) if is_pc else np.array([s1], dtype=np.float16)
        payload['p2'] = np.packbits(bits2, bitorder='little')
        payload['s2'] = s2.astype(np.float16) if is_pc else np.array([s2], dtype=np.float16)
    elif mode == 'sparse_res_3pct':
        m = np.abs(arr) > 0.7 * np.abs(arr).mean()
        s = float(np.abs(arr[m]).mean()) if m.any() else 1.0
        q1 = s * np.sign(arr) * m
        res = arr - q1
        k = max(1, int(arr.size * 0.03))
        cutoff = np.partition(np.abs(res).ravel(), -k)[-k]
        res_m = np.abs(res) >= cutoff
        coords = np.argwhere(res_m).astype(np.uint16)
        vals = res[res_m].astype(np.float16)
        codes = np.where(m, np.sign(arr), 0).astype(np.int8).ravel() + 1
        bits = np.stack([(codes >> 1) & 1, codes & 1], 1).ravel()
        payload['bits'] = np.packbits(bits, bitorder='little')
        payload['scale'] = np.array([s], dtype=np.float16)
        payload['coords'] = coords
        payload['vals'] = vals
    elif mode == 'lowrank_r4':
        m = np.abs(arr) > 0.7 * np.abs(arr).mean()
        s = float(np.abs(arr[m]).mean()) if m.any() else 1.0
        q1 = s * np.sign(arr) * m
        res = arr - q1
        U, S, Vh = np.linalg.svd(res, full_matrices=False)
        r = min(4, len(S))
        U_r = (U[:, :r] * np.sqrt(S[:r])).astype(np.float16)
        V_r = (Vh[:r, :].T * np.sqrt(S[:r])).astype(np.float16)
        codes = np.where(m, np.sign(arr), 0).astype(np.int8).ravel() + 1
        bits = np.stack([(codes >> 1) & 1, codes & 1], 1).ravel()
        payload['bits'] = np.packbits(bits, bitorder='little')
        payload['scale'] = np.array([s], dtype=np.float16)
        payload['U'] = U_r
        payload['V'] = V_r
    elif mode == 'degraded_control':
        m = np.abs(arr) > np.median(np.abs(arr))
        bits = np.packbits(m.ravel(), bitorder='little')
        payload['bits'] = bits
    else:
        payload['val'] = arr.astype(np.float16)
    return payload

def measure_transformer_artifact_bytes(model, quant_modes, candidate_name):
    bundle = {}
    meta = {'name': candidate_name, 'quant_modes': quant_modes, 'total_params': model.get_num_params()}
    
    for name, p in model.named_parameters():
        arr = p.detach().cpu().numpy()
        if 'wte' in name: mode = quant_modes.get('emb', 'fp16')
        elif 'wpe' in name: mode = quant_modes.get('pos', 'fp16')
        elif 'attn.q_proj' in name: mode = quant_modes.get('attn_q', 'fp16')
        elif 'attn.k_proj' in name: mode = quant_modes.get('attn_k', 'fp16')
        elif 'attn.v_proj' in name: mode = quant_modes.get('attn_v', 'fp16')
        elif 'attn.out_proj' in name: mode = quant_modes.get('attn_out', 'fp16')
        elif 'mlp_fc1' in name: mode = quant_modes.get('mlp_fc1', 'fp16')
        elif 'mlp_fc2' in name: mode = quant_modes.get('mlp_fc2', 'fp16')
        elif 'lm_head' in name: mode = quant_modes.get('head', 'fp16')
        else: mode = 'fp16'
        
        packed = pack_matrix_to_bytes(arr, mode)
        for k, v in packed.items():
            bundle[f"{name}.{k}"] = v
        meta[f"{name}.shape"] = list(arr.shape)
        meta[f"{name}.mode"] = mode
        
    out_dir = ROOT / 'checkpoints/artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{candidate_name}.npz"
    json_path = out_dir / f"{candidate_name}.json"
    
    np.savez_compressed(npz_path, **bundle)
    json_path.write_text(json.dumps(meta, indent=2))
    total_bytes = npz_path.stat().st_size + json_path.stat().st_size
    return total_bytes

def benchmark_inference(model, quant_modes):
    model.eval()
    model.set_quant_modes(quant_modes)
    
    prompt = torch.randint(0, 256, (1, 64), device=DEVICE)
    with torch.no_grad():
        for _ in range(5): _ = model(prompt)
    if DEVICE == 'mps': torch.mps.synchronize()
    
    prefill_times = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            _ = model(prompt)
            if DEVICE == 'mps': torch.mps.synchronize()
            prefill_times.append((time.perf_counter() - t0) * 1e3)
    p50_prefill_ms = float(np.median(prefill_times))
    
    decode_times = []
    with torch.no_grad():
        ctx = prompt.clone()
        for _ in range(30):
            t0 = time.perf_counter()
            logits, _ = model(ctx)
            next_t = torch.argmax(logits[:, -1:, :], dim=-1)
            ctx = torch.cat([ctx[:, 1:], next_t], dim=1)
            if DEVICE == 'mps': torch.mps.synchronize()
            decode_times.append((time.perf_counter() - t0) * 1e6)
    p50_decode_us = float(np.median(decode_times))
    tput_tok_per_sec = 1e6 / p50_decode_us
    
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    return p50_prefill_ms, p50_decode_us, tput_tok_per_sec, peak_rss_mb

def run_suite():
    model, val_loader, test_loader = load_or_train_reference(epochs=3, lr=1e-3, seed=42)
    
    ref_ce, ref_ppl = evaluate_perplexity(model, test_loader, {})
    ref_bytes = measure_transformer_artifact_bytes(model, {}, 'FP16_Reference')
    ref_prefill, ref_decode, ref_tput, ref_ram = benchmark_inference(model, {})
    
    print(f"\n=== REFERENCE FP16 TRANSFORMER (1.17M Parameters) ===")
    print(f"  Test Loss: {ref_ce:.4f} nats | Test Perplexity: {ref_ppl:.3f}")
    print(f"  Total Artifact Bytes: {ref_bytes:,} B ({ref_bytes/(1024*1024):.2f} MB)")
    print(f"  Prefill (T=64): {ref_prefill:.2f} ms | Decode: {ref_decode:.1f} us/tok ({ref_tput:.1f} tok/s) | RAM: {ref_ram:.1f} MB\n")
    
    candidates = [
        ("FP16_Reference", {}),
        ("Degraded_Negative_Control", {k: 'degraded_control' for k in ['emb','pos','attn_q','attn_k','attn_v','attn_out','mlp_fc1','mlp_fc2','head']}),
        
        # A. Full Ternary Quantization
        ("Full_Ternary_Global", {k: 'ternary_global' for k in ['emb','pos','attn_q','attn_k','attn_v','attn_out','mlp_fc1','mlp_fc2','head']}),
        ("Full_Ternary_Per_Channel", {k: 'ternary_pc' for k in ['emb','pos','attn_q','attn_k','attn_v','attn_out','mlp_fc1','mlp_fc2','head']}),
        
        # B & C. Dual-Basis Quantization
        ("Global_Dual_Basis_All", {k: 'dbq_global' for k in ['emb','pos','attn_q','attn_k','attn_v','attn_out','mlp_fc1','mlp_fc2','head']}),
        ("PC_Dual_Basis_All", {k: 'dbq_pc' for k in ['emb','pos','attn_q','attn_k','attn_v','attn_out','mlp_fc1','mlp_fc2','head']}),
        
        # Component Sensitivity Ablations
        ("Sens_Quant_Attn_Only", {'attn_q': 'dbq_pc', 'attn_k': 'dbq_pc', 'attn_v': 'dbq_pc', 'attn_out': 'dbq_pc'}),
        ("Sens_Quant_MLP_Only", {'mlp_fc1': 'dbq_pc', 'mlp_fc2': 'dbq_pc'}),
        ("Sens_Quant_Emb_Head_Only", {'emb': 'dbq_pc', 'pos': 'dbq_pc', 'head': 'dbq_pc'}),
        
        # D. Adaptive Layer-Wise Precision (A-T-DBQ)
        ("Adaptive_Transformer_DBQ_V1", {'emb': 'dbq_pc', 'pos': 'dbq_pc', 'attn_q': 'fp16', 'attn_k': 'fp16', 'attn_v': 'dbq_pc', 'attn_out': 'dbq_pc', 'mlp_fc1': 'dbq_pc', 'mlp_fc2': 'dbq_pc', 'head': 'dbq_pc'}),
        ("Adaptive_Transformer_DBQ_V2", {'emb': 'dbq_pc', 'pos': 'dbq_pc', 'attn_q': 'fp16', 'attn_k': 'fp16', 'attn_v': 'fp16', 'attn_out': 'fp16', 'mlp_fc1': 'dbq_pc', 'mlp_fc2': 'dbq_pc', 'head': 'dbq_pc'}),
        
        # E. Sparse Residual Correction
        ("Ternary_Sparse_Residual_3pct", {k: 'sparse_res_3pct' for k in ['emb','pos','attn_q','attn_k','attn_v','attn_out','mlp_fc1','mlp_fc2','head']}),
        
        # F. Low-Rank Residual Correction
        ("Ternary_Low_Rank_r4", {k: 'lowrank_r4' for k in ['emb','pos','attn_q','attn_k','attn_v','attn_out','mlp_fc1','mlp_fc2','head']}),
    ]
    
    results = []
    print("=" * 125)
    print(f"{'Candidate Name':32s} | {'Test PPL':8s} | {'PPL Gap':8s} | {'Artifact (B)':12s} | {'Comp Ratio':10s} | {'Decode us':10s} | {'Tok/s':8s} | {'RAM MB':8s}")
    print("=" * 125)
    
    for name, modes in candidates:
        ce, ppl = evaluate_perplexity(model, test_loader, modes)
        gap_pct = (ppl - ref_ppl) / ref_ppl * 100.0
        art_bytes = measure_transformer_artifact_bytes(model, modes, name)
        comp_ratio = ref_bytes / art_bytes if art_bytes > 0 else 1.0
        prefill_ms, decode_us, tput, ram_mb = benchmark_inference(model, modes)
        
        row = {
            "candidate_name": name,
            "test_ce_nats": round(ce, 4),
            "test_perplexity": round(ppl, 3),
            "ppl_gap_pct": round(gap_pct, 2),
            "artifact_bytes": art_bytes,
            "compression_ratio": round(comp_ratio, 2),
            "prefill_p50_ms": round(prefill_ms, 2),
            "decode_p50_us": round(decode_us, 1),
            "throughput_tok_per_sec": round(tput, 1),
            "peak_rss_ram_mb": round(ram_mb, 1)
        }
        results.append(row)
        print(f"{name:32s} | {ppl:8.3f} | {gap_pct:+7.2f}% | {art_bytes:10,d} B | {comp_ratio:9.2f}x | {decode_us:8.1f} us | {tput:6.1f} | {ram_mb:6.1f}")
        
    print("=" * 125)
    
    out_csv = ROOT / 'evidence/transformer_compression_benchmark.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nBenchmark results saved to: {out_csv}")
    
    print("\n--- Qualitative 64-Byte Autoregressive Text Generation Samples ---")
    model.eval()
    prompt = torch.tensor([[ord(c) for c in "The European lobster is "]], dtype=torch.long, device=DEVICE)
    for sample_cand in ["FP16_Reference", "Degraded_Negative_Control", "Full_Ternary_Global", "PC_Dual_Basis_All", "Adaptive_Transformer_DBQ_V2"]:
        cand_modes = dict(candidates)[sample_cand]
        model.set_quant_modes(cand_modes)
        out_tokens = model.generate(prompt, max_new_tokens=64)
        out_str = bytes(out_tokens[0].cpu().tolist()).decode('latin1', errors='replace')
        print(f"[{sample_cand:30s}]: {repr(out_str)}")

if __name__ == '__main__':
    run_suite()
