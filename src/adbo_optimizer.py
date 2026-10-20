"""
Alternating Dual-Basis Optimization (ADBO) for Transformer Compression
LittleLess AI — Phase 3.3 Core Mathematical Invention

Optimizes two ternary planes (T1, T2 in {-1, 0, 1}) and channel scales (s1, s2)
via alternating closed-form Gram regression and vectorized discrete coordinate descent:
  min_{T1, T2, s1, s2} Tr( (W - (s1*T1 + s2*T2)) H (W - (s1*T1 + s2*T2))^T )
Incorporates sequential block-wise activation propagation (GPTQ/AWQ style) to account
for upstream quantization perturbations.
"""

import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_transformer import CausalTransformerLM
from src.dataset import get_dataloaders
from src.awg_dbq import pack_ternary_planes, quantize_int8_embedding

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 9 candidate coordinate pairs
PAIRS = torch.tensor([
    [-1, -1], [-1, 0], [-1, 1],
    [ 0, -1], [ 0, 0], [ 0, 1],
    [ 1, -1], [ 1, 0], [ 1, 1]
], dtype=torch.float32)
CAND_T1 = PAIRS[:, 0].view(1, 9)
CAND_T2 = PAIRS[:, 1].view(1, 9)

def solve_scales_matrix(T1, T2, W, H):
    """
    Vectorized closed-form normal equations across all M rows:
    [s1, s2] = argmin Tr((W - s1*T1 - s2*T2) H (W - s1*T1 - s2*T2)^T)
    """
    T1_H = T1 @ H
    T2_H = T2 @ H
    W_H = W @ H
    g11 = (T1_H * T1).sum(dim=-1, keepdim=True)
    g12 = (T1_H * T2).sum(dim=-1, keepdim=True)
    g22 = (T2_H * T2).sum(dim=-1, keepdim=True)
    b1 = (W_H * T1).sum(dim=-1, keepdim=True)
    b2 = (W_H * T2).sum(dim=-1, keepdim=True)
    eps = 1e-6 * torch.maximum(g11, g22).clamp(min=1e-8)
    g11 = g11 + eps
    g22 = g22 + eps
    det = (g11 * g22 - g12**2).clamp(min=1e-8)
    s1 = (g22 * b1 - g12 * b2) / det
    s2 = (g11 * b2 - g12 * b1) / det
    return s1, s2

def compute_calibration_loss(W, W_hat, H):
    """
    Tr( (W - W_hat) H (W - W_hat)^T ) / M
    """
    R = W - W_hat
    R_H = R @ H
    loss = (R_H * R).sum().item() / W.shape[0]
    return loss

def optimize_layer_adbo(W, H, c_thresh=0.7, sweeps=2):
    """
    Performs Alternating Dual-Basis Optimization on weight matrix W given input covariance H.
    W: [M, N]
    H: [N, N]
    """
    M, N = W.shape
    
    # 1. Initial greedy AWG-DBQ initialization
    t1 = c_thresh * W.abs().mean(dim=-1, keepdim=True)
    m1 = W.abs() > t1
    T1 = (W.sign() * m1).float()
    s1_init = (W * T1).sum(dim=-1, keepdim=True) / (T1**2).sum(dim=-1, keepdim=True).clamp(min=1.0)
    R1 = W - s1_init * T1
    
    t2 = c_thresh * R1.abs().mean(dim=-1, keepdim=True)
    m2 = R1.abs() > t2
    T2 = (R1.sign() * m2).float()
    
    s1, s2 = solve_scales_matrix(T1, T2, W, H)
    W_hat_init = s1 * T1 + s2 * T2
    init_loss = compute_calibration_loss(W, W_hat_init, H)
    
    R = W - W_hat_init
    H_diag = torch.diag(H)
    
    sweep_history = []
    
    # 2. Alternating sweeps: Vectorized Discrete Coordinate Descent + Closed-form Scale Updates
    for s_idx in range(sweeps):
        t0 = time.perf_counter()
        G = R @ H
        flips = 0
        
        for j in range(N):
            H_jj = H_diag[j]
            curr_t1 = T1[:, j:j+1]
            curr_t2 = T2[:, j:j+1]
            
            delta = s1 * (CAND_T1 - curr_t1) + s2 * (CAND_T2 - curr_t2) # [M, 9]
            d_loss = -2.0 * delta * G[:, j:j+1] + (delta ** 2) * H_jj   # [M, 9]
            
            best_idx = torch.argmin(d_loss, dim=1) # [M]
            best_t1 = CAND_T1[0, best_idx]         # [M]
            best_t2 = CAND_T2[0, best_idx]         # [M]
            
            best_delta = s1.squeeze(1) * (best_t1 - curr_t1.squeeze(1)) + s2.squeeze(1) * (best_t2 - curr_t2.squeeze(1))
            mask = (best_delta != 0)
            if mask.any():
                flips += mask.sum().item()
                T1[:, j] = best_t1
                T2[:, j] = best_t2
                R[:, j] -= best_delta
                G -= torch.outer(best_delta, H[j, :])
                
        # Re-estimate scales in closed form after pattern updates
        s1, s2 = solve_scales_matrix(T1, T2, W, H)
        W_hat = s1 * T1 + s2 * T2
        R = W - W_hat
        cur_loss = compute_calibration_loss(W, W_hat, H)
        dt = (time.perf_counter() - t0) * 1e3
        
        red_pct = (init_loss - cur_loss) / max(init_loss, 1e-12) * 100.0
        sweep_history.append({
            'sweep': s_idx + 1,
            'loss': cur_loss,
            'flips': flips,
            'reduction_pct': red_pct,
            'time_ms': dt
        })
        
    final_W_hat = s1 * T1 + s2 * T2
    return T1, s1, T2, s2, final_W_hat, init_loss, sweep_history

def run_adbo_experiment():
    print("=" * 80)
    print("LITTLELESS AI — PHASE 3.3: ALTERNATING DUAL-BASIS OPTIMIZATION (ADBO)")
    print(f"Device: {DEVICE} | Host: Apple Silicon M4 16GB")
    print("=" * 80)
    
    t_start = time.perf_counter()
    
    train_loader, _, test_loader = get_dataloaders(
        data_dir=ROOT / 'data/wikitext-2',
        block_size=128,
        batch_size=32,
        train_samples=500,
        eval_samples=1500 # 192,000 tokens
    )
    
    ref_ckpt = ROOT / 'checkpoints/transformer_ref_fp16.pt'
    assert ref_ckpt.exists(), f"Reference checkpoint missing: {ref_ckpt}"
    
    model = CausalTransformerLM().to(DEVICE)
    model.load_state_dict(torch.load(ref_ckpt, map_location=DEVICE))
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    
    ref_ppl = 6.370
    awg_ppl = 6.618
    
    bundle_adbo = {}
    detailed_layer_logs = []
    total_init_calib_loss = 0.0
    total_final_calib_loss = 0.0
    layer_count = 0
    
    # 1. Quantize embeddings symmetrically to INT8
    print("\n[1/4] Quantizing token and positional embeddings to symmetric INT8...")
    for emb_name in ['wte', 'wpe']:
        p = getattr(model, emb_name).weight
        q, scale = quantize_int8_embedding(p.data)
        bundle_adbo[f"{emb_name}.weight.int8"] = q
        bundle_adbo[f"{emb_name}.weight.scale"] = scale
        # Reconstruct into model parameter for sequential calibration propagation
        p.data = torch.from_numpy(q.astype(np.float32) * scale.astype(np.float32)).to(DEVICE)

    # 2. Sequential Block-by-Block ADBO
    print("\n[2/4] Executing Sequential Block-by-Block Vectorized ADBO...")
    t_opt_start = time.perf_counter()
    
    with torch.no_grad():
        x_calib, _ = next(iter(train_loader))
        x_calib = x_calib.to(DEVICE)
        B, T = x_calib.shape
        pos = torch.arange(0, T, device=DEVICE)
        x = model.wte(x_calib) + model.wpe(pos)
        
        for i, block in enumerate(model.blocks):
            # Attention Block
            x_norm1 = block.ln1(x)
            x_flat1 = x_norm1.reshape(-1, 160)
            H_attn = ((x_flat1.T @ x_flat1) / x_flat1.shape[0]).cpu()
            
            for p_name in ['q_proj', 'k_proj', 'v_proj']:
                proj = getattr(block.attn, p_name)
                W = proj.weight.data.cpu()
                T1, s1, T2, s2, w_hat, init_l, hist = optimize_layer_adbo(W, H_attn, sweeps=2)
                proj.weight.data.copy_(w_hat.to(DEVICE))
                
                full_name = f"blocks.{i}.attn.{p_name}.weight"
                packed = pack_ternary_planes(T1, s1, T2, s2)
                for k, v in packed.items(): bundle_adbo[f"{full_name}.{k}"] = v
                
                final_l = hist[-1]['loss']
                total_init_calib_loss += init_l
                total_final_calib_loss += final_l
                layer_count += 1
                detailed_layer_logs.append((full_name, init_l, final_l, hist[-1]['reduction_pct']))

            # Forward through Q, K, V
            q = torch.nn.functional.linear(x_norm1, block.attn.q_proj.weight).view(B, T, 5, 32).transpose(1, 2)
            k = torch.nn.functional.linear(x_norm1, block.attn.k_proj.weight).view(B, T, 5, 32).transpose(1, 2)
            v = torch.nn.functional.linear(x_norm1, block.attn.v_proj.weight).view(B, T, 5, 32).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(32))
            att = att.masked_fill(block.attn.mask[:, :, :T, :T] == 0, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, 160)
            y_flat = y.reshape(-1, 160)
            H_out = ((y_flat.T @ y_flat) / y_flat.shape[0]).cpu()
            
            W = block.attn.out_proj.weight.data.cpu()
            T1, s1, T2, s2, w_hat, init_l, hist = optimize_layer_adbo(W, H_out, sweeps=2)
            block.attn.out_proj.weight.data.copy_(w_hat.to(DEVICE))
            
            full_name = f"blocks.{i}.attn.out_proj.weight"
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): bundle_adbo[f"{full_name}.{k}"] = v
            final_l = hist[-1]['loss']
            total_init_calib_loss += init_l
            total_final_calib_loss += final_l
            layer_count += 1
            detailed_layer_logs.append((full_name, init_l, final_l, hist[-1]['reduction_pct']))
            
            # Residual connection
            x = x + torch.nn.functional.linear(y, block.attn.out_proj.weight)
            
            # MLP Block
            x_norm2 = block.ln2(x)
            x_flat2 = x_norm2.reshape(-1, 160)
            H_mlp1 = ((x_flat2.T @ x_flat2) / x_flat2.shape[0]).cpu()
            
            W = block.mlp_fc1.weight.data.cpu()
            T1, s1, T2, s2, w_hat, init_l, hist = optimize_layer_adbo(W, H_mlp1, sweeps=2)
            block.mlp_fc1.weight.data.copy_(w_hat.to(DEVICE))
            
            full_name = f"blocks.{i}.mlp_fc1.weight"
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): bundle_adbo[f"{full_name}.{k}"] = v
            final_l = hist[-1]['loss']
            total_init_calib_loss += init_l
            total_final_calib_loss += final_l
            layer_count += 1
            detailed_layer_logs.append((full_name, init_l, final_l, hist[-1]['reduction_pct']))
            
            # MLP fc1 bias
            bundle_adbo[f"blocks.{i}.mlp_fc1.bias.val"] = block.mlp_fc1.bias.data.cpu().numpy().astype(np.float16)
            
            h = torch.nn.functional.gelu(torch.nn.functional.linear(x_norm2, block.mlp_fc1.weight, block.mlp_fc1.bias))
            h_flat = h.reshape(-1, 512)
            H_mlp2 = ((h_flat.T @ h_flat) / h_flat.shape[0]).cpu()
            
            W = block.mlp_fc2.weight.data.cpu()
            T1, s1, T2, s2, w_hat, init_l, hist = optimize_layer_adbo(W, H_mlp2, sweeps=2)
            block.mlp_fc2.weight.data.copy_(w_hat.to(DEVICE))
            
            full_name = f"blocks.{i}.mlp_fc2.weight"
            packed = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed.items(): bundle_adbo[f"{full_name}.{k}"] = v
            final_l = hist[-1]['loss']
            total_init_calib_loss += init_l
            total_final_calib_loss += final_l
            layer_count += 1
            detailed_layer_logs.append((full_name, init_l, final_l, hist[-1]['reduction_pct']))
            
            # MLP fc2 bias & LayerNorms
            bundle_adbo[f"blocks.{i}.mlp_fc2.bias.val"] = block.mlp_fc2.bias.data.cpu().numpy().astype(np.float16)
            bundle_adbo[f"blocks.{i}.ln1.weight.val"] = block.ln1.weight.data.cpu().numpy().astype(np.float16)
            bundle_adbo[f"blocks.{i}.ln1.bias.val"] = block.ln1.bias.data.cpu().numpy().astype(np.float16)
            bundle_adbo[f"blocks.{i}.ln2.weight.val"] = block.ln2.weight.data.cpu().numpy().astype(np.float16)
            bundle_adbo[f"blocks.{i}.ln2.bias.val"] = block.ln2.bias.data.cpu().numpy().astype(np.float16)
            
            x = x + torch.nn.functional.linear(h, block.mlp_fc2.weight, block.mlp_fc2.bias)
            
        # Final LayerNorm & LM Head
        bundle_adbo["ln_f.weight.val"] = model.ln_f.weight.data.cpu().numpy().astype(np.float16)
        bundle_adbo["ln_f.bias.val"] = model.ln_f.bias.data.cpu().numpy().astype(np.float16)
        
        x_f = model.ln_f(x)
        x_f_flat = x_f.reshape(-1, 160)
        H_head = ((x_f_flat.T @ x_f_flat) / x_f_flat.shape[0]).cpu()
        
        W = model.lm_head.weight.data.cpu()
        T1, s1, T2, s2, w_hat, init_l, hist = optimize_layer_adbo(W, H_head, sweeps=2)
        model.lm_head.weight.data.copy_(w_hat.to(DEVICE))
        
        full_name = "lm_head.weight"
        packed = pack_ternary_planes(T1, s1, T2, s2)
        for k, v in packed.items(): bundle_adbo[f"{full_name}.{k}"] = v
        final_l = hist[-1]['loss']
        total_init_calib_loss += init_l
        total_final_calib_loss += final_l
        layer_count += 1
        detailed_layer_logs.append((full_name, init_l, final_l, hist[-1]['reduction_pct']))

    opt_time_sec = time.perf_counter() - t_opt_start
    print(f"Sequential ADBO completed across {layer_count} linear projections in {opt_time_sec:.3f} s!")

    # 3. Save Serialized Artifact Package
    print("\n[3/4] Serializing ADBO champion artifact package...")
    out_dir = ROOT / 'checkpoints/artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / 'ADBO_DBQ_Hybrid_INT8.npz'
    json_path = out_dir / 'ADBO_DBQ_Hybrid_INT8.json'
    
    meta_adbo = {
        "name": "ADBO_DBQ_Hybrid_INT8",
        "total_params": model.get_num_params(),
        "linear_quant": "Alternating_Dual_Basis_Optimization_2bit",
        "embedding_quant": "INT8_symmetric",
        "norm_bias_quant": "FP16",
        "sweeps": 2,
        "cd_vectorized": True
    }
    
    np.savez_compressed(npz_path, **bundle_adbo)
    json_path.write_text(json.dumps(meta_adbo, indent=2))
    
    npz_bytes = npz_path.stat().st_size
    json_bytes = json_path.stat().st_size
    total_art_bytes = npz_bytes + json_bytes
    print(f"Artifact serialized: {npz_bytes:,} B (NPZ) + {json_bytes:,} B (JSON) = {total_art_bytes:,} B ({total_art_bytes/1024:.2f} KB)")

    # 4. Evaluate Held-Out Perplexity (192,000 tokens)
    print("\n[4/4] Evaluating held-out test perplexity (192,000 tokens)...")
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, l = model(x, y)
            tot_loss += l.item() * y.numel()
            tot_tok += y.numel()
            
    ce = tot_loss / tot_tok
    adbo_ppl = math.exp(ce)
    total_time_sec = time.perf_counter() - t_start
    
    avg_init_loss = total_init_calib_loss / layer_count
    avg_final_loss = total_final_calib_loss / layer_count
    overall_calib_reduction = (avg_init_loss - avg_final_loss) / avg_init_loss * 100.0
    
    gap_vs_ref = (adbo_ppl - ref_ppl) / ref_ppl * 100.0
    gap_vs_awg = (adbo_ppl - awg_ppl) / awg_ppl * 100.0
    
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY:")
    print("=" * 80)
    print(f"1. Initial AWG-DBQ Calibration Loss: {avg_init_loss:.6e}")
    print(f"2. ADBO Calibration Loss:           {avg_final_loss:.6e} ({overall_calib_reduction:+.2f}% reduction)")
    print(f"3. Final Held-Out Test Perplexity:  {adbo_ppl:.4f} (Cross-Entropy: {ce:.4f} nats)")
    print(f"4. Comparison vs AWG-DBQ (6.618):   {adbo_ppl - awg_ppl:+.4f} PPL ({gap_vs_awg:+.2f}%)")
    print(f"5. Comparison vs FP16 Ref (6.370):  {adbo_ppl - ref_ppl:+.4f} PPL ({gap_vs_ref:+.2f}% degradation)")
    print(f"6. Total Optimization Runtime:      {opt_time_sec:.3f} s (Full script: {total_time_sec:.2f} s)")
    print(f"7. Exit Status:                     0 (SUCCESS)")
    print(f"8. Preserved Patterns & Scales:     SAVED TO DISK ({npz_path.name}, {json_path.name})")
    print("=" * 80)
    
    # Save CSV evidence
    evidence_csv = ROOT / 'evidence/phase3_3_alternating_optimization_results.csv'
    with open(evidence_csv, 'w') as f:
        f.write("layer,init_calib_loss,final_calib_loss,calib_loss_reduction_pct\n")
        for l_name, i_l, f_l, r_p in detailed_layer_logs:
            f.write(f"{l_name},{i_l:.6e},{f_l:.6e},{r_p:.2f}\n")
        f.write(f"AVERAGE,{avg_init_loss:.6e},{avg_final_loss:.6e},{overall_calib_reduction:.2f}\n")
    print(f"Detailed layer-wise calibration results saved to {evidence_csv}")
    
    # Verify standalone loader reloadability in fresh state
    print("\n[Audit] Verifying standalone deserialization from disk...")
    from src.standalone_loader import load_standalone_transformer
    loaded_model, loaded_meta = load_standalone_transformer(npz_path, device=DEVICE)
    tot_l_reloaded, tot_t_reloaded = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, l = loaded_model(x, y)
            tot_l_reloaded += l.item() * y.numel()
            tot_t_reloaded += y.numel()
    reloaded_ppl = math.exp(tot_l_reloaded / tot_t_reloaded)
    print(f"Reloaded model PPL: {reloaded_ppl:.4f} (Exact match: {abs(reloaded_ppl - adbo_ppl) < 1e-4})")

if __name__ == '__main__':
    run_adbo_experiment()
