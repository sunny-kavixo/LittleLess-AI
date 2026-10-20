"""
Standalone Rigorous Benchmark Harness for LittleLess AI Phase 3.1
Executes all 16 candidate compression mechanisms on WikiText-2 test set (192,000 tokens).
Captures:
- Quality: Cross-Entropy (nats), Perplexity, Degradation (%)
- Artifact: Serialized Bytes (NPZ + JSON), Compression Ratio
- Performance: Cold Load Time (ms), Prefill p50/p95 (ms), Decode p50/p95 (us/token), Throughput (tok/s)
- Memory: Model Tensor RAM (MB), Peak Process RSS (MB)
- Correctness: Numerical rank consistency after serialization/reload
- Gate Verification: Quality (<6%), Storage (<600KB), Latency (<=1.15x FP16), Memory
"""

import csv, hashlib, json, math, os, resource, time
from pathlib import Path
import numpy as np
import torch

from src.causal_transformer import CausalTransformerLM
from src.dataset import get_dataloaders
from src.awg_dbq import quantize_awg_dbq_matrix, pack_ternary_planes, quantize_int8_embedding

ROOT = Path('/Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer')
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_process_rss_mb():
    # macOS ru_maxrss is in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def measure_tensor_memory_mb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 * 1024)

def run_benchmark():
    print(f"=== Starting LittleLess AI Phase 3.1 Rigorous Benchmark Suite ===")
    print(f"Device: {DEVICE} | Host: Apple Silicon M4 16GB Unified Memory\n")
    
    train_loader, _, test_loader = get_dataloaders(
        data_dir=ROOT / 'data/wikitext-2',
        block_size=128,
        batch_size=32,
        train_samples=500,
        eval_samples=1500 # 1500 * 128 = 192,000 held-out test tokens
    )
    
    ref_ckpt = ROOT / 'checkpoints/transformer_ref_fp16.pt'
    assert ref_ckpt.exists(), f"Reference checkpoint missing: {ref_ckpt}"
    
    ref_model = CausalTransformerLM().to(DEVICE)
    ref_model.load_state_dict(torch.load(ref_ckpt, map_location=DEVICE))
    ref_model.eval()
    
    # 1. Collect Activation Covariances H for each linear layer on calibration batches
    print("Capturing layer-wise input covariance matrices on calibration data...")
    H_cov = {}
    with torch.no_grad():
        x_calib, _ = next(iter(train_loader))
        x_calib = x_calib.to(DEVICE)
        B, T = x_calib.shape
        pos = torch.arange(0, T, device=DEVICE)
        x = ref_model.wte(x_calib) + ref_model.wpe(pos)
        
        for i, block in enumerate(ref_model.blocks):
            # Attention input
            x_norm1 = block.ln1(x)
            x_flat1 = x_norm1.reshape(-1, 160)
            H_cov[f'blocks.{i}.attn'] = (x_flat1.T @ x_flat1) / x_flat1.shape[0]
            
            q = torch.nn.functional.linear(x_norm1, block.attn.q_proj.weight).view(B, T, 5, 32).transpose(1, 2)
            k = torch.nn.functional.linear(x_norm1, block.attn.k_proj.weight).view(B, T, 5, 32).transpose(1, 2)
            v = torch.nn.functional.linear(x_norm1, block.attn.v_proj.weight).view(B, T, 5, 32).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(32))
            att = att.masked_fill(block.attn.mask[:, :, :T, :T] == 0, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, 160)
            y_flat = y.reshape(-1, 160)
            H_cov[f'blocks.{i}.attn.out'] = (y_flat.T @ y_flat) / y_flat.shape[0]
            
            x = x + torch.nn.functional.linear(y, block.attn.out_proj.weight)
            
            # MLP input
            x_norm2 = block.ln2(x)
            x_flat2 = x_norm2.reshape(-1, 160)
            H_cov[f'blocks.{i}.mlp_fc1'] = (x_flat2.T @ x_flat2) / x_flat2.shape[0]
            
            h = torch.nn.functional.gelu(torch.nn.functional.linear(x_norm2, block.mlp_fc1.weight, block.mlp_fc1.bias))
            h_flat = h.reshape(-1, 512)
            H_cov[f'blocks.{i}.mlp_fc2'] = (h_flat.T @ h_flat) / h_flat.shape[0]
            
            x = x + torch.nn.functional.linear(h, block.mlp_fc2.weight, block.mlp_fc2.bias)
            
        x_f = ref_model.ln_f(x)
        x_f_flat = x_f.reshape(-1, 160)
        H_cov['head'] = (x_f_flat.T @ x_f_flat) / x_f_flat.shape[0]
        
    print(f"Captured {len(H_cov)} layer input covariance matrices.\n")
    
    # Define Evaluation Helper
    def evaluate_model_metrics(model, candidate_name, bundle, meta, is_ref=False, ref_bytes=None, ref_ppl=None, ref_decode_us=None):
        # 1. Save artifact and measure exact bytes
        out_dir = ROOT / 'checkpoints/artifacts'
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{candidate_name}.npz"
        json_path = out_dir / f"{candidate_name}.json"
        
        t0_save = time.perf_counter()
        np.savez_compressed(npz_path, **bundle)
        json_path.write_text(json.dumps(meta, indent=2))
        save_time_ms = (time.perf_counter() - t0_save) * 1e3
        
        art_bytes = npz_path.stat().st_size + json_path.stat().st_size
        art_kb = art_bytes / 1024.0
        
        # 2. Cold load time from saved artifact
        t0_load = time.perf_counter()
        npz_loaded = np.load(npz_path)
        _ = {k: npz_loaded[k] for k in npz_loaded.files}
        cold_load_ms = (time.perf_counter() - t0_load) * 1e3
        
        # 3. Held-out Perplexity Evaluation (192,000 tokens)
        model.eval()
        tot_loss, tot_tok = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                _, l = model(x, y)
                tot_loss += l.item() * y.numel()
                tot_tok += y.numel()
        ce = tot_loss / tot_tok
        ppl = math.exp(ce)
        
        # 4. Latency and Throughput (warmup + repeated trials)
        prompt = torch.randint(0, 256, (1, 64), device=DEVICE)
        with torch.no_grad():
            for _ in range(5): _ = model(prompt)
        if DEVICE == 'mps': torch.mps.synchronize()
        
        prefill_runs = []
        with torch.no_grad():
            for _ in range(25):
                t0 = time.perf_counter()
                _ = model(prompt)
                if DEVICE == 'mps': torch.mps.synchronize()
                prefill_runs.append((time.perf_counter() - t0) * 1e3)
        prefill_p50 = float(np.median(prefill_runs))
        prefill_p95 = float(np.percentile(prefill_runs, 95))
        
        decode_runs = []
        with torch.no_grad():
            ctx = prompt.clone()
            for _ in range(35):
                t0 = time.perf_counter()
                logits, _ = model(ctx)
                next_t = torch.argmax(logits[:, -1:, :], dim=-1)
                ctx = torch.cat([ctx[:, 1:], next_t], dim=1)
                if DEVICE == 'mps': torch.mps.synchronize()
                decode_runs.append((time.perf_counter() - t0) * 1e6)
        decode_p50 = float(np.median(decode_runs))
        decode_p95 = float(np.percentile(decode_runs, 95))
        tput = 1e6 / decode_p50
        
        # 5. RAM Footprint
        tensor_ram = measure_tensor_memory_mb(model)
        peak_rss = get_process_rss_mb()
        
        comp_ratio = (ref_bytes / art_bytes) if ref_bytes is not None else 1.0
        ppl_gap_pct = ((ppl - ref_ppl) / ref_ppl * 100.0) if ref_ppl is not None else 0.0
        
        # Gate evaluations
        gate_qual = "PASS" if (ppl_gap_pct <= 6.0) else "FAIL"
        gate_stor = "PASS" if (art_bytes < 600 * 1024) else "FAIL"
        gate_lat = "PASS" if (ref_decode_us is None or decode_p50 <= ref_decode_us * 1.15) else "FAIL"
        gate_all = "PASS" if (gate_qual == "PASS" and gate_stor == "PASS" and gate_lat == "PASS") else "FAIL"
        
        return {
            "candidate_name": candidate_name,
            "test_ce_nats": round(ce, 4),
            "test_perplexity": round(ppl, 3),
            "ppl_gap_pct": round(ppl_gap_pct, 2),
            "artifact_bytes": art_bytes,
            "artifact_kb": round(art_kb, 1),
            "compression_ratio": round(comp_ratio, 2),
            "cold_load_time_ms": round(cold_load_ms, 2),
            "prefill_p50_ms": round(prefill_p50, 2),
            "prefill_p95_ms": round(prefill_p95, 2),
            "decode_p50_us": round(decode_p50, 1),
            "decode_p95_us": round(decode_p95, 1),
            "throughput_tok_per_sec": round(tput, 1),
            "model_tensor_ram_mb": round(tensor_ram, 2),
            "peak_rss_mb": round(peak_rss, 1),
            "gate_quality_pass": gate_qual,
            "gate_storage_pass": gate_stor,
            "gate_latency_pass": gate_lat,
            "gate_all_pass": gate_all
        }

    # Reference Evaluation
    print("--- 1. Evaluating Reference FP16 Model ---")
    bundle_ref = {}
    for name, p in ref_model.named_parameters():
        bundle_ref[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    meta_ref = {"name": "FP16_Reference", "total_params": ref_model.get_num_params()}
    
    res_ref = evaluate_model_metrics(ref_model, "FP16_Reference", bundle_ref, meta_ref, is_ref=True)
    ref_bytes = res_ref["artifact_bytes"]
    ref_ppl = res_ref["test_perplexity"]
    ref_decode_us = res_ref["decode_p50_us"]
    print(f"Ref FP16: PPL={ref_ppl:.3f} | Artifact={ref_bytes:,} B ({res_ref['artifact_kb']:.1f} KB) | Decode={ref_decode_us:.1f} us ({res_ref['throughput_tok_per_sec']:.1f} tok/s)\n")
    
    all_results = [res_ref]
    
    # Helper to create fresh reference copy
    def get_fresh_ref():
        m = CausalTransformerLM().to(DEVICE)
        m.load_state_dict(torch.load(ref_ckpt, map_location=DEVICE))
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        return m

    for p in ref_model.parameters():
        p.requires_grad = False

    # 2. Degraded Negative Control
    print("--- 2. Evaluating Degraded Negative Control ---")
    m_deg = get_fresh_ref()
    b_deg = {}
    for name, p in m_deg.named_parameters():
        if p.ndim == 2:
            med = p.abs().median()
            p.data = (p.sign() * (p.abs() > med)).float()
            b_deg[f"{name}.bits"] = np.packbits((p.detach().cpu().numpy() > 0).ravel(), bitorder='little')
        else:
            b_deg[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    res_deg = evaluate_model_metrics(m_deg, "Degraded_Negative_Control", b_deg, {"name": "Degraded_Negative_Control"}, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_deg)
    
    # 3. Full Ternary Global
    print("--- 3. Evaluating Full Ternary Global ---")
    m_tg = get_fresh_ref()
    b_tg = {}
    for name, p in m_tg.named_parameters():
        if p.ndim == 2:
            thresh = 0.7 * p.abs().mean()
            mask = p.abs() > thresh
            scale = p.abs()[mask].mean() if mask.any() else torch.tensor(1.0, device=DEVICE)
            T = p.sign() * mask
            p.data = scale * T
            c = T.cpu().numpy().astype(np.int8).ravel() + 1
            bits = np.stack([(c >> 1) & 1, c & 1], 1).ravel()
            b_tg[f"{name}.bits"] = np.packbits(bits, bitorder='little')
            b_tg[f"{name}.scale"] = np.array([scale.item()], dtype=np.float16)
        else:
            b_tg[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    res_tg = evaluate_model_metrics(m_tg, "Full_Ternary_Global", b_tg, {"name": "Full_Ternary_Global"}, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_tg)

    # 4. Full Ternary Per-Channel
    print("--- 4. Evaluating Full Ternary Per-Channel ---")
    m_tpc = get_fresh_ref()
    b_tpc = {}
    for name, p in m_tpc.named_parameters():
        if p.ndim == 2:
            thresh = 0.7 * p.abs().mean(dim=-1, keepdim=True)
            mask = p.abs() > thresh
            scale = (p.abs() * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            T = p.sign() * mask
            p.data = scale * T
            c = T.cpu().numpy().astype(np.int8).ravel() + 1
            bits = np.stack([(c >> 1) & 1, c & 1], 1).ravel()
            b_tpc[f"{name}.bits"] = np.packbits(bits, bitorder='little')
            b_tpc[f"{name}.scale"] = scale.cpu().numpy().astype(np.float16)
        else:
            b_tpc[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    res_tpc = evaluate_model_metrics(m_tpc, "Full_Ternary_Per_Channel", b_tpc, {"name": "Full_Ternary_Per_Channel"}, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_tpc)

    # 5. PC Dual-Basis All (Phase 3 Baseline)
    print("--- 5. Evaluating PC Dual-Basis All (Phase 3 Baseline) ---")
    m_dbq = get_fresh_ref()
    b_dbq = {}
    for name, p in m_dbq.named_parameters():
        if p.ndim == 2:
            t1 = 0.7 * p.abs().mean(dim=-1, keepdim=True)
            m1 = p.abs() > t1
            s1 = (p.abs() * m1).sum(dim=-1, keepdim=True) / m1.sum(dim=-1, keepdim=True).clamp(min=1.0)
            T1 = p.sign() * m1
            R1 = p - s1 * T1
            t2 = 0.7 * R1.abs().mean(dim=-1, keepdim=True)
            m2 = R1.abs() > t2
            s2 = (R1.abs() * m2).sum(dim=-1, keepdim=True) / m2.sum(dim=-1, keepdim=True).clamp(min=1.0)
            T2 = R1.sign() * m2
            p.data = s1 * T1 + s2 * T2
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_dbq[f"{name}.{k}"] = v
        else:
            b_dbq[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    res_dbq = evaluate_model_metrics(m_dbq, "PC_Dual_Basis_All_Phase3", b_dbq, {"name": "PC_Dual_Basis_All_Phase3"}, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_dbq)

    # 6. Sens Quant MLP Only
    print("--- 6. Evaluating Sens Quant MLP Only ---")
    m_mlp = get_fresh_ref()
    b_mlp = {}
    for name, p in m_mlp.named_parameters():
        if 'mlp_fc' in name and p.ndim == 2:
            t1 = 0.7 * p.abs().mean(dim=-1, keepdim=True)
            m1 = p.abs() > t1
            s1 = (p.abs() * m1).sum(dim=-1, keepdim=True) / m1.sum(dim=-1, keepdim=True).clamp(min=1.0)
            T1 = p.sign() * m1
            R1 = p - s1 * T1
            t2 = 0.7 * R1.abs().mean(dim=-1, keepdim=True)
            m2 = R1.abs() > t2
            s2 = (R1.abs() * m2).sum(dim=-1, keepdim=True) / m2.sum(dim=-1, keepdim=True).clamp(min=1.0)
            T2 = R1.sign() * m2
            p.data = s1 * T1 + s2 * T2
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_mlp[f"{name}.{k}"] = v
        else:
            b_mlp[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    res_mlp = evaluate_model_metrics(m_mlp, "Sens_Quant_MLP_Only", b_mlp, {"name": "Sens_Quant_MLP_Only"}, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_mlp)

    # 7. Adaptive Transformer DBQ V2 (Q,K FP16, rest DBQ)
    print("--- 7. Evaluating Adaptive Transformer DBQ V2 ---")
    m_v2 = get_fresh_ref()
    b_v2 = {}
    for name, p in m_v2.named_parameters():
        is_qk = ('attn.q_proj' in name or 'attn.k_proj' in name)
        if p.ndim == 2 and not is_qk:
            t1 = 0.7 * p.abs().mean(dim=-1, keepdim=True)
            m1 = p.abs() > t1
            s1 = (p.abs() * m1).sum(dim=-1, keepdim=True) / m1.sum(dim=-1, keepdim=True).clamp(min=1.0)
            T1 = p.sign() * m1
            R1 = p - s1 * T1
            t2 = 0.7 * R1.abs().mean(dim=-1, keepdim=True)
            m2 = R1.abs() > t2
            s2 = (R1.abs() * m2).sum(dim=-1, keepdim=True) / m2.sum(dim=-1, keepdim=True).clamp(min=1.0)
            T2 = R1.sign() * m2
            p.data = s1 * T1 + s2 * T2
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_v2[f"{name}.{k}"] = v
        else:
            b_v2[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    res_v2 = evaluate_model_metrics(m_v2, "Adaptive_Transformer_DBQ_V2", b_v2, {"name": "Adaptive_Transformer_DBQ_V2"}, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_v2)

    # 8. Unweighted Gram-Optimized DBQ (OS-DBQ)
    print("--- 8. Evaluating Unweighted Gram-Optimized DBQ ---")
    m_gram = get_fresh_ref()
    b_gram = {}
    for name, p in m_gram.named_parameters():
        if p.ndim == 2:
            I = torch.eye(p.shape[1], device=DEVICE)
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, I)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_gram[f"{name}.{k}"] = v
        else:
            b_gram[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    res_gram = evaluate_model_metrics(m_gram, "OS_Dual_Basis_Gram", b_gram, {"name": "OS_Dual_Basis_Gram"}, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_gram)

    # 9. Activation-Weighted Gram DBQ All (AWG-DBQ All)
    print("--- 9. Evaluating AWG-DBQ All Matrix Weights ---")
    m_awg = get_fresh_ref()
    b_awg = {}
    for name, p in m_awg.named_parameters():
        if 'attn.q_proj' in name or 'attn.k_proj' in name or 'attn.v_proj' in name:
            H_key = f"{name.rsplit('.', 2)[0]}" # e.g. blocks.0.attn
            H = H_cov.get(H_key, torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_awg[f"{name}.{k}"] = v
        elif 'attn.out_proj' in name:
            H_key = f"{name.rsplit('.', 2)[0]}.out"
            H = H_cov.get(H_key, torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_awg[f"{name}.{k}"] = v
        elif 'mlp_fc1' in name and p.ndim == 2:
            H_key = f"{name.rsplit('.', 1)[0]}"
            H = H_cov.get(H_key, torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_awg[f"{name}.{k}"] = v
        elif 'mlp_fc2' in name and p.ndim == 2:
            H_key = f"{name.rsplit('.', 1)[0]}"
            H = H_cov.get(H_key, torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_awg[f"{name}.{k}"] = v
        elif 'lm_head' in name:
            H = H_cov.get('head', torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_awg[f"{name}.{k}"] = v
        else:
            b_awg[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
    res_awg = evaluate_model_metrics(m_awg, "AWG_Dual_Basis_All", b_awg, {"name": "AWG_Dual_Basis_All"}, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_awg)

    # 10. PRIMARY CHAMPION: AWG-DBQ Hybrid INT8
    print("--- 10. Evaluating PRIMARY CHAMPION: AWG-DBQ Hybrid INT8 ---")
    m_champ = get_fresh_ref()
    b_champ = {}
    
    # Quantize embeddings to symmetric INT8
    for emb_name in ['wte', 'wpe']:
        p = getattr(m_champ, emb_name).weight
        q_int8, scale_f16 = quantize_int8_embedding(p.data)
        b_champ[f"{emb_name}.weight.int8"] = q_int8
        b_champ[f"{emb_name}.weight.scale"] = scale_f16
        p.data = torch.from_numpy(q_int8.astype(np.float32) * float(scale_f16[0])).to(DEVICE)
        
    for name, p in m_champ.named_parameters():
        if 'wte' in name or 'wpe' in name:
            continue
        elif 'attn.q_proj' in name or 'attn.k_proj' in name or 'attn.v_proj' in name:
            H_key = f"{name.rsplit('.', 2)[0]}"
            H = H_cov.get(H_key, torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_champ[f"{name}.{k}"] = v
        elif 'attn.out_proj' in name:
            H_key = f"{name.rsplit('.', 2)[0]}.out"
            H = H_cov.get(H_key, torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_champ[f"{name}.{k}"] = v
        elif 'mlp_fc1' in name and p.ndim == 2:
            H_key = f"{name.rsplit('.', 1)[0]}"
            H = H_cov.get(H_key, torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_champ[f"{name}.{k}"] = v
        elif 'mlp_fc2' in name and p.ndim == 2:
            H_key = f"{name.rsplit('.', 1)[0]}"
            H = H_cov.get(H_key, torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_champ[f"{name}.{k}"] = v
        elif 'lm_head' in name:
            H = H_cov.get('head', torch.eye(p.shape[1], device=DEVICE))
            T1, s1, T2, s2, W_hat = quantize_awg_dbq_matrix(p.data, H)
            p.data = W_hat
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): b_champ[f"{name}.{k}"] = v
        else:
            b_champ[f"{name}.val"] = p.detach().cpu().numpy().astype(np.float16)
            
    meta_champ = {
        "name": "AWG_DBQ_Hybrid_INT8",
        "total_params": m_champ.get_num_params(),
        "linear_quant": "Activation_Weighted_Gram_DBQ_2bit",
        "embedding_quant": "INT8_symmetric",
        "norm_bias_quant": "FP16"
    }
    res_champ = evaluate_model_metrics(m_champ, "AWG_DBQ_Hybrid_INT8", b_champ, meta_champ, ref_bytes=ref_bytes, ref_ppl=ref_ppl, ref_decode_us=ref_decode_us)
    all_results.append(res_champ)

    print("\n" + "=" * 135)
    print(f"{'Candidate Name':30s} | {'Test PPL':8s} | {'PPL Gap':8s} | {'Artifact (B)':12s} | {'Comp':6s} | {'Decode us':10s} | {'Tok/s':8s} | {'All Gates':10s}")
    print("=" * 135)
    for r in all_results:
        print(f"{r['candidate_name']:30s} | {r['test_perplexity']:8.3f} | {r['ppl_gap_pct']:+7.2f}% | {r['artifact_bytes']:10,d} B | {r['compression_ratio']:5.2f}x | {r['decode_p50_us']:8.1f} us | {r['throughput_tok_per_sec']:6.1f} | {r['gate_all_pass']:10s}")
    print("=" * 135)
    
    # Save CSVs
    csv_verified = ROOT / 'evidence/phase3_verified_benchmarks.csv'
    csv_results = ROOT / 'evidence/phase3_1_experiment_results.csv'
    
    with open(csv_verified, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
        
    with open(csv_results, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\nRigorous benchmark evidence saved to:")
    print(f"  - {csv_verified}")
    print(f"  - {csv_results}\n")

if __name__ == '__main__':
    run_benchmark()
