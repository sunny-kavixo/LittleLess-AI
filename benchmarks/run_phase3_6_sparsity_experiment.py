"""
LittleLess AI — Phase 3.6 Sparsity-Aware Compression Experiment Driver
Controlled experiment investigating Sparsity-Aware ADBO (S-ADBO) across 8 candidates:
  - C0 (Control): Unregularized ADBO (lambda = 0.0)
  - S1: S-ADBO (lambda = 0.001)
  - S2: S-ADBO (lambda = 0.003)
  - S3: S-ADBO (lambda = 0.006)
  - S4: S-ADBO (lambda = 0.012)
  - S5: S-ADBO (lambda = 0.025)
  - B1: Naive magnitude pruning baseline (c = 0.8)
  - B2: Naive magnitude pruning baseline (c = 1.2)
"""

import copy, ctypes, gc, gzip, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Users/sunny/.gemini/antigravity-ide/scratch/LittleLess-AI-Phase3-Transformer')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_transformer import CausalTransformerLM
from src.dataset import get_dataloaders
from src.awg_dbq import pack_ternary_planes, quantize_int8_embedding
from src.radix3_packing import pack_radix3_layer
from src.sparsity_adbo_optimizer import (
    optimize_layer_sadbo,
    naive_magnitude_prune_layer,
    compute_trit_statistics
)

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load native C++ library
LIB_PATH = ROOT / 'src/kernels/libnative_transformer.dylib'
assert LIB_PATH.exists(), f"Native library missing: {LIB_PATH}"
lib = ctypes.CDLL(str(LIB_PATH))

# Set C-FFI types
lib.create_native_packed_model.argtypes = [ctypes.c_int]
lib.create_native_packed_model.restype = ctypes.c_void_p
lib.free_native_packed_model.argtypes = [ctypes.c_void_p]

lib.set_embeddings.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int8), ctypes.c_float,
    ctypes.POINTER(ctypes.c_int8), ctypes.c_float
]

lib.set_block_norms.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]

lib.set_block_linear.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
]

lib.set_final_head.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int
]

lib.native_generate_tokens.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)
]
lib.native_generate_tokens.restype = ctypes.c_int


def load_native_model_from_bundle(bundle, pack_mode=0):
    model = lib.create_native_packed_model(ctypes.c_int(pack_mode))
    
    wte_q = np.ascontiguousarray(bundle['wte.weight.int8'], dtype=np.int8)
    wte_s = float(bundle['wte.weight.scale'][0])
    wpe_q = np.ascontiguousarray(bundle['wpe.weight.int8'], dtype=np.int8)
    wpe_s = float(bundle['wpe.weight.scale'][0])

    lib.set_embeddings(
        model,
        wte_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), ctypes.c_float(wte_s),
        wpe_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), ctypes.c_float(wpe_s)
    )

    p1_key_suffix = '.p1_radix3' if pack_mode == 1 else '.p1'
    p2_key_suffix = '.p2_radix3' if pack_mode == 1 else '.p2'

    for b_idx in range(4):
        ln1_w = np.ascontiguousarray(bundle[f'blocks.{b_idx}.ln1.weight.val'].astype(np.float32))
        ln1_b = np.ascontiguousarray(bundle[f'blocks.{b_idx}.ln1.bias.val'].astype(np.float32))
        ln2_w = np.ascontiguousarray(bundle[f'blocks.{b_idx}.ln2.weight.val'].astype(np.float32))
        ln2_b = np.ascontiguousarray(bundle[f'blocks.{b_idx}.ln2.bias.val'].astype(np.float32))
        
        lib.set_block_norms(
            model, ctypes.c_int(b_idx),
            ln1_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ln1_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ln2_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ln2_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        proj_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'mlp_fc1', 'mlp_fc2']
        for l_type, p_name in enumerate(proj_names):
            prefix = f'blocks.{b_idx}.{p_name}' if 'mlp' in p_name else f'blocks.{b_idx}.attn.{p_name}'
            p1 = np.ascontiguousarray(bundle[f'{prefix}.weight{p1_key_suffix}'], dtype=np.uint8)
            p2 = np.ascontiguousarray(bundle[f'{prefix}.weight{p2_key_suffix}'], dtype=np.uint8)
            s1 = np.ascontiguousarray(bundle[f'{prefix}.weight.s1'].astype(np.float32))
            s2 = np.ascontiguousarray(bundle[f'{prefix}.weight.s2'].astype(np.float32))
            
            bias = None
            if f'{prefix}.bias.val' in bundle:
                bias = np.ascontiguousarray(bundle[f'{prefix}.bias.val'].astype(np.float32))
                
            M = 512 if p_name == 'mlp_fc1' else 160
            N = 512 if p_name == 'mlp_fc2' else 160
            
            lib.set_block_linear(
                model, ctypes.c_int(b_idx), ctypes.c_int(l_type),
                p1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                p2.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                s1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if bias is not None else None,
                ctypes.c_int(M), ctypes.c_int(N)
            )

    ln_f_w = np.ascontiguousarray(bundle['ln_f.weight.val'].astype(np.float32))
    ln_f_b = np.ascontiguousarray(bundle['ln_f.bias.val'].astype(np.float32))
    head_p1 = np.ascontiguousarray(bundle[f'lm_head.weight{p1_key_suffix}'], dtype=np.uint8)
    head_p2 = np.ascontiguousarray(bundle[f'lm_head.weight{p2_key_suffix}'], dtype=np.uint8)
    head_s1 = np.ascontiguousarray(bundle['lm_head.weight.s1'].astype(np.float32))
    head_s2 = np.ascontiguousarray(bundle['lm_head.weight.s2'].astype(np.float32))

    lib.set_final_head(
        model,
        ln_f_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ln_f_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        head_p1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        head_p2.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        head_s1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        head_s2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(256), ctypes.c_int(160)
    )
    return model


def benchmark_native_model(model_handle, num_trials=35, prompt_len=4, gen_tokens=32):
    prompt = np.array([42, 65, 99, 12], dtype=np.int32)
    out_tokens = np.zeros(gen_tokens, dtype=np.int32)
    stats = np.zeros(4, dtype=np.float64)
    
    # Warmup
    for _ in range(5):
        lib.native_generate_tokens(
            model_handle,
            prompt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int(prompt_len),
            ctypes.c_int(gen_tokens),
            out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
    prefill_ms_list = []
    decode_ms_list = []
    
    for _ in range(num_trials):
        lib.native_generate_tokens(
            model_handle,
            prompt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int(prompt_len),
            ctypes.c_int(gen_tokens),
            out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        prefill_ms_list.append(stats[0])
        decode_ms_list.append(stats[1])
        
    p50_pre = float(np.median(prefill_ms_list))
    p95_pre = float(np.percentile(prefill_ms_list, 95))
    p50_dec = float(np.median(decode_ms_list)) * 1e3 / gen_tokens # us per token
    p95_dec = float(np.percentile(decode_ms_list, 95)) * 1e3 / gen_tokens
    tput = 1e6 / p50_dec
    
    return {
        'prefill_p50_ms': p50_pre,
        'prefill_p95_ms': p95_pre,
        'decode_p50_us': p50_dec,
        'decode_p95_us': p95_dec,
        'throughput_tok_s': tput,
        'tokens': [int(t) for t in out_tokens[:10]]
    }


def evaluate_perplexity(model, test_loader):
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, l = model(x, y)
            tot_loss += l.item() * y.numel()
            tot_tok += y.numel()
    ce = tot_loss / tot_tok
    return math.exp(ce), ce


def run_single_candidate(candidate_id, cand_type, param_val, ref_ckpt, train_loader, test_loader, c0_cache=None):
    """
    Executes optimization, serialization, trit stats, PPL, and native benchmarks for one candidate.
    """
    print(f"\n>>> Running Candidate {candidate_id} [{cand_type} | param={param_val}] <<<")
    t0 = time.perf_counter()
    
    model = CausalTransformerLM().to(DEVICE)
    model.load_state_dict(torch.load(ref_ckpt, map_location=DEVICE))
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    
    bundle = {}
    t1_list = []
    t2_list = []
    
    # 1. Quantize embeddings
    for emb_name in ['wte', 'wpe']:
        p = getattr(model, emb_name).weight
        q, scale = quantize_int8_embedding(p.data)
        bundle[f"{emb_name}.weight.int8"] = q
        bundle[f"{emb_name}.weight.scale"] = scale
        p.data = torch.from_numpy(q.astype(np.float32) * scale.astype(np.float32)).to(DEVICE)
        
    # 2. Block-by-block optimization
    with torch.no_grad():
        x_calib, _ = next(iter(train_loader))
        x_calib = x_calib.to(DEVICE)
        B, T = x_calib.shape
        pos = torch.arange(0, T, device=DEVICE)
        x = model.wte(x_calib) + model.wpe(pos)
        
        for i, block in enumerate(model.blocks):
            # Attention
            x_norm1 = block.ln1(x)
            x_flat1 = x_norm1.reshape(-1, 160)
            H_attn = ((x_flat1.T @ x_flat1) / x_flat1.shape[0]).cpu()
            
            for p_name in ['q_proj', 'k_proj', 'v_proj']:
                proj = getattr(block.attn, p_name)
                W = proj.weight.data.cpu()
                
                if cand_type == 'S_ADBO':
                    T1, s1, T2, s2, w_hat, _, _ = optimize_layer_sadbo(W, H_attn, sweeps=2, sparsity_lambda=param_val)
                elif cand_type == 'PRUNE':
                    # Retrieve C0 unregularized T1, T2
                    c0_T1, c0_s1, c0_T2, c0_s2 = c0_cache[f"blocks.{i}.attn.{p_name}"]
                    T1, s1, T2, s2, w_hat = naive_magnitude_prune_layer(c0_T1, c0_s1, c0_T2, c0_s2, W, thresh_factor=param_val)
                    
                t1_list.append(T1)
                t2_list.append(T2)
                proj.weight.data.copy_(w_hat.to(DEVICE))
                
                full_name = f"blocks.{i}.attn.{p_name}.weight"
                packed_2b = pack_ternary_planes(T1, s1, T2, s2)
                for k, v in packed_2b.items(): bundle[f"{full_name}.{k}"] = v
                packed_r3 = pack_radix3_layer(T1, s1, T2, s2)
                bundle[f"{full_name}.p1_radix3"] = packed_r3['p1_radix3']
                bundle[f"{full_name}.p2_radix3"] = packed_r3['p2_radix3']
                
            # Forward Attention QKV
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
            if cand_type == 'S_ADBO':
                T1, s1, T2, s2, w_hat, _, _ = optimize_layer_sadbo(W, H_out, sweeps=2, sparsity_lambda=param_val)
            elif cand_type == 'PRUNE':
                c0_T1, c0_s1, c0_T2, c0_s2 = c0_cache[f"blocks.{i}.attn.out_proj"]
                T1, s1, T2, s2, w_hat = naive_magnitude_prune_layer(c0_T1, c0_s1, c0_T2, c0_s2, W, thresh_factor=param_val)
                
            t1_list.append(T1)
            t2_list.append(T2)
            block.attn.out_proj.weight.data.copy_(w_hat.to(DEVICE))
            
            full_name = f"blocks.{i}.attn.out_proj.weight"
            packed_2b = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed_2b.items(): bundle[f"{full_name}.{k}"] = v
            packed_r3 = pack_radix3_layer(T1, s1, T2, s2)
            bundle[f"{full_name}.p1_radix3"] = packed_r3['p1_radix3']
            bundle[f"{full_name}.p2_radix3"] = packed_r3['p2_radix3']
            
            x = x + torch.nn.functional.linear(y, block.attn.out_proj.weight)
            
            # MLP
            x_norm2 = block.ln2(x)
            x_flat2 = x_norm2.reshape(-1, 160)
            H_mlp1 = ((x_flat2.T @ x_flat2) / x_flat2.shape[0]).cpu()
            
            W = block.mlp_fc1.weight.data.cpu()
            if cand_type == 'S_ADBO':
                T1, s1, T2, s2, w_hat, _, _ = optimize_layer_sadbo(W, H_mlp1, sweeps=2, sparsity_lambda=param_val)
            elif cand_type == 'PRUNE':
                c0_T1, c0_s1, c0_T2, c0_s2 = c0_cache[f"blocks.{i}.mlp_fc1"]
                T1, s1, T2, s2, w_hat = naive_magnitude_prune_layer(c0_T1, c0_s1, c0_T2, c0_s2, W, thresh_factor=param_val)
                
            t1_list.append(T1)
            t2_list.append(T2)
            block.mlp_fc1.weight.data.copy_(w_hat.to(DEVICE))
            
            full_name = f"blocks.{i}.mlp_fc1.weight"
            packed_2b = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed_2b.items(): bundle[f"{full_name}.{k}"] = v
            packed_r3 = pack_radix3_layer(T1, s1, T2, s2)
            bundle[f"{full_name}.p1_radix3"] = packed_r3['p1_radix3']
            bundle[f"{full_name}.p2_radix3"] = packed_r3['p2_radix3']
            bundle[f"blocks.{i}.mlp_fc1.bias.val"] = block.mlp_fc1.bias.data.cpu().numpy().astype(np.float16)
            
            h = torch.nn.functional.gelu(torch.nn.functional.linear(x_norm2, block.mlp_fc1.weight, block.mlp_fc1.bias))
            h_flat = h.reshape(-1, 512)
            H_mlp2 = ((h_flat.T @ h_flat) / h_flat.shape[0]).cpu()
            
            W = block.mlp_fc2.weight.data.cpu()
            if cand_type == 'S_ADBO':
                T1, s1, T2, s2, w_hat, _, _ = optimize_layer_sadbo(W, H_mlp2, sweeps=2, sparsity_lambda=param_val)
            elif cand_type == 'PRUNE':
                c0_T1, c0_s1, c0_T2, c0_s2 = c0_cache[f"blocks.{i}.mlp_fc2"]
                T1, s1, T2, s2, w_hat = naive_magnitude_prune_layer(c0_T1, c0_s1, c0_T2, c0_s2, W, thresh_factor=param_val)
                
            t1_list.append(T1)
            t2_list.append(T2)
            block.mlp_fc2.weight.data.copy_(w_hat.to(DEVICE))
            
            full_name = f"blocks.{i}.mlp_fc2.weight"
            packed_2b = pack_ternary_planes(T1, s1, T2, s2)
            for k, v in packed_2b.items(): bundle[f"{full_name}.{k}"] = v
            packed_r3 = pack_radix3_layer(T1, s1, T2, s2)
            bundle[f"{full_name}.p1_radix3"] = packed_r3['p1_radix3']
            bundle[f"{full_name}.p2_radix3"] = packed_r3['p2_radix3']
            
            bundle[f"blocks.{i}.mlp_fc2.bias.val"] = block.mlp_fc2.bias.data.cpu().numpy().astype(np.float16)
            bundle[f"blocks.{i}.ln1.weight.val"] = block.ln1.weight.data.cpu().numpy().astype(np.float16)
            bundle[f"blocks.{i}.ln1.bias.val"] = block.ln1.bias.data.cpu().numpy().astype(np.float16)
            bundle[f"blocks.{i}.ln2.weight.val"] = block.ln2.weight.data.cpu().numpy().astype(np.float16)
            bundle[f"blocks.{i}.ln2.bias.val"] = block.ln2.bias.data.cpu().numpy().astype(np.float16)
            
            x = x + torch.nn.functional.linear(h, block.mlp_fc2.weight, block.mlp_fc2.bias)
            
        # Final LM Head
        bundle["ln_f.weight.val"] = model.ln_f.weight.data.cpu().numpy().astype(np.float16)
        bundle["ln_f.bias.val"] = model.ln_f.bias.data.cpu().numpy().astype(np.float16)
        
        x_f = model.ln_f(x)
        x_f_flat = x_f.reshape(-1, 160)
        H_head = ((x_f_flat.T @ x_f_flat) / x_f_flat.shape[0]).cpu()
        
        W = model.lm_head.weight.data.cpu()
        if cand_type == 'S_ADBO':
            T1, s1, T2, s2, w_hat, _, _ = optimize_layer_sadbo(W, H_head, sweeps=2, sparsity_lambda=param_val)
        elif cand_type == 'PRUNE':
            c0_T1, c0_s1, c0_T2, c0_s2 = c0_cache["lm_head"]
            T1, s1, T2, s2, w_hat = naive_magnitude_prune_layer(c0_T1, c0_s1, c0_T2, c0_s2, W, thresh_factor=param_val)
            
        t1_list.append(T1)
        t2_list.append(T2)
        model.lm_head.weight.data.copy_(w_hat.to(DEVICE))
        
        full_name = "lm_head.weight"
        packed_2b = pack_ternary_planes(T1, s1, T2, s2)
        for k, v in packed_2b.items(): bundle[f"{full_name}.{k}"] = v
        packed_r3 = pack_radix3_layer(T1, s1, T2, s2)
        bundle[f"{full_name}.p1_radix3"] = packed_r3['p1_radix3']
        bundle[f"{full_name}.p2_radix3"] = packed_r3['p2_radix3']

    # 3. Compute Trit Statistics & Entropy
    trit_stats = compute_trit_statistics(t1_list, t2_list)
    
    # 4. Save Artifacts
    out_dir = ROOT / 'checkpoints/artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2-bit NPZ
    npz_2b_path = out_dir / f"S_ADBO_{candidate_id}_2bit.npz"
    json_2b_path = out_dir / f"S_ADBO_{candidate_id}_2bit.json"
    bundle_2b = {k: v for k, v in bundle.items() if not k.endswith('_radix3')}
    np.savez_compressed(npz_2b_path, **bundle_2b)
    meta = {
        'candidate_id': candidate_id,
        'type': cand_type,
        'param': param_val,
        'zero_pct': trit_stats['zero_pct'],
        'entropy_bits_per_trit': trit_stats['entropy_bits_per_trit']
    }
    json_2b_path.write_text(json.dumps(meta, indent=2))
    size_2b = npz_2b_path.stat().st_size + json_2b_path.stat().st_size
    
    # Radix-3 NPZ
    npz_r3_path = out_dir / f"S_ADBO_{candidate_id}_Radix3.npz"
    json_r3_path = out_dir / f"S_ADBO_{candidate_id}_Radix3.json"
    bundle_r3 = {k: v for k, v in bundle.items() if not (k.endswith('.p1') or k.endswith('.p2'))}
    np.savez_compressed(npz_r3_path, **bundle_r3)
    json_r3_path.write_text(json.dumps(meta, indent=2))
    size_r3 = npz_r3_path.stat().st_size + json_r3_path.stat().st_size
    
    # Raw Gzip stream of ternary planes
    raw_ternary = b''.join([bundle[k].tobytes() for k in sorted(bundle_2b.keys()) if k.endswith('.p1') or k.endswith('.p2')])
    gzip_bytes = len(gzip.compress(raw_ternary, compresslevel=9))
    
    # 5. Evaluate Perplexity
    test_ppl, test_ce = evaluate_perplexity(model, test_loader)
    
    # 6. Benchmark Native 2-Bit NEON Engine
    m_native_2b = load_native_model_from_bundle(bundle, pack_mode=0)
    perf_2b = benchmark_native_model(m_native_2b)
    lib.free_native_packed_model(m_native_2b)
    
    # 7. Benchmark Native Radix-3 Engine
    m_native_r3 = load_native_model_from_bundle(bundle, pack_mode=1)
    perf_r3 = benchmark_native_model(m_native_r3)
    lib.free_native_packed_model(m_native_r3)
    
    dt_total = time.perf_counter() - t0
    
    res = {
        'candidate_id': candidate_id,
        'type': cand_type,
        'param': param_val,
        'zero_pct': trit_stats['zero_pct'],
        'p_neg': trit_stats['p_neg'],
        'p_zero': trit_stats['p_zero'],
        'p_pos': trit_stats['p_pos'],
        'entropy': trit_stats['entropy_bits_per_trit'],
        'shannon_bound_bytes': trit_stats['theoretical_ternary_bytes'],
        'size_2b_npz_bytes': size_2b,
        'size_r3_npz_bytes': size_r3,
        'size_gzip_planes_bytes': gzip_bytes,
        'weight_ram_2b_bytes': 680712,
        'weight_ram_r3_bytes': 570140,
        'test_ppl': test_ppl,
        'test_ce': test_ce,
        'prefill_p50_ms': perf_2b['prefill_p50_ms'],
        'decode_p50_us': perf_2b['decode_p50_us'],
        'throughput_2b': perf_2b['throughput_tok_s'],
        'throughput_r3': perf_r3['throughput_tok_s'],
        'tokens': perf_2b['tokens'],
        'time_sec': dt_total
    }
    
    # Cache layer patterns if this is C0
    c0_layer_cache = {}
    if candidate_id == 'C0':
        idx = 0
        for i in range(4):
            for p_name in ['q_proj', 'k_proj', 'v_proj']:
                c0_layer_cache[f"blocks.{i}.attn.{p_name}"] = (t1_list[idx], bundle[f"blocks.{i}.attn.{p_name}.weight.s1"], t2_list[idx], bundle[f"blocks.{i}.attn.{p_name}.weight.s2"])
                idx += 1
            c0_layer_cache[f"blocks.{i}.attn.out_proj"] = (t1_list[idx], bundle[f"blocks.{i}.attn.out_proj.weight.s1"], t2_list[idx], bundle[f"blocks.{i}.attn.out_proj.weight.s2"])
            idx += 1
            c0_layer_cache[f"blocks.{i}.mlp_fc1"] = (t1_list[idx], bundle[f"blocks.{i}.mlp_fc1.weight.s1"], t2_list[idx], bundle[f"blocks.{i}.mlp_fc1.weight.s2"])
            idx += 1
            c0_layer_cache[f"blocks.{i}.mlp_fc2"] = (t1_list[idx], bundle[f"blocks.{i}.mlp_fc2.weight.s1"], t2_list[idx], bundle[f"blocks.{i}.mlp_fc2.weight.s2"])
            idx += 1
        c0_layer_cache["lm_head"] = (t1_list[idx], bundle["lm_head.weight.s1"], t2_list[idx], bundle["lm_head.weight.s2"])
        
    print(f"  Zeros: {res['zero_pct']:.2f}% | Entropy: {res['entropy']:.4f} b/trit | PPL: {res['test_ppl']:.4f} | 2B NPZ: {res['size_2b_npz_bytes']:,} B | 2B Tok/s: {res['throughput_2b']:.1f}")
    return res, c0_layer_cache


def run_phase3_6_experiment():
    print("=" * 110)
    print("LITTLELESS AI — PHASE 3.6: SPARSITY-AWARE TERNARY COMPRESSION CONTROLLED EXPERIMENT")
    print(f"Device: {DEVICE} | Host: Apple Silicon M4 16GB Unified Memory")
    print("=" * 110)
    
    t_exp_start = time.perf_counter()
    ref_ckpt = ROOT / 'checkpoints/transformer_ref_fp16.pt'
    assert ref_ckpt.exists(), f"Reference checkpoint missing: {ref_ckpt}"
    
    train_loader, _, test_loader = get_dataloaders(
        data_dir=ROOT / 'data/wikitext-2',
        block_size=128,
        batch_size=32,
        train_samples=500,
        eval_samples=1500 # 192,000 tokens
    )
    
    candidates = [
        ('C0', 'S_ADBO', 0.0),    # Control: unregularized ADBO
        ('S1', 'S_ADBO', 0.001),  # Low penalty
        ('S2', 'S_ADBO', 0.003),  # Low-mid penalty
        ('S3', 'S_ADBO', 0.006),  # Mid penalty
        ('S4', 'S_ADBO', 0.012),  # Mid-high penalty
        ('S5', 'S_ADBO', 0.025),  # High penalty
        ('B1', 'PRUNE', 0.8),     # Naive prune baseline 1
        ('B2', 'PRUNE', 1.2),     # Naive prune baseline 2
    ]
    
    results = []
    c0_cache = None
    
    for c_id, c_type, param in candidates:
        res, cache = run_single_candidate(c_id, c_type, param, ref_ckpt, train_loader, test_loader, c0_cache=c0_cache)
        results.append(res)
        if c_id == 'C0':
            c0_cache = cache
            
    # Multi-seed stability test on C0 and best Pareto candidate (e.g. S3 or S2)
    print("\n" + "=" * 110)
    print("MULTI-SEED REPRODUCIBILITY VERIFICATION (Seeds 42, 1337, 2026)")
    print("=" * 110)
    seeds = [42, 1337, 2026]
    seed_results = []
    
    # Pick S2 and S3 for multi-seed verification
    for target_cand, target_lambda in [('C0', 0.0), ('S3', 0.006)]:
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            res_s, _ = run_single_candidate(f"{target_cand}_s{s}", 'S_ADBO', target_lambda, ref_ckpt, train_loader, test_loader)
            seed_results.append(res_s)
            
    # Write master CSV evidence
    evidence_csv = ROOT / 'evidence/phase3_6_sparsity_benchmarks.csv'
    with open(evidence_csv, 'w') as f:
        f.write("candidate_id,type,param,zero_pct,p_neg,p_zero,p_pos,entropy_bits_per_trit,shannon_bound_bytes,size_2b_npz_bytes,size_r3_npz_bytes,size_gzip_planes_bytes,weight_ram_2b_bytes,weight_ram_r3_bytes,test_ppl,test_ce,prefill_p50_ms,decode_p50_us,throughput_2b,throughput_r3,time_sec\n")
        for r in results + seed_results:
            f.write(f"{r['candidate_id']},{r['type']},{r['param']},{r['zero_pct']:.4f},{r['p_neg']:.4f},{r['p_zero']:.4f},{r['p_pos']:.4f},{r['entropy']:.4f},{r['shannon_bound_bytes']:.1f},{r['size_2b_npz_bytes']},{r['size_r3_npz_bytes']},{r['size_gzip_planes_bytes']},{r['weight_ram_2b_bytes']},{r['weight_ram_r3_bytes']},{r['test_ppl']:.4f},{r['test_ce']:.4f},{r['prefill_p50_ms']:.2f},{r['decode_p50_us']:.1f},{r['throughput_2b']:.1f},{r['throughput_r3']:.1f},{r['time_sec']:.2f}\n")
    print(f"\nAll benchmark evidence successfully saved to: {evidence_csv}")
    
    # Print formatted comparative table
    print("\n" + "=" * 135)
    print(f"{'Candidate':10s} | {'Type':7s} | {'Param':7s} | {'Zero %':8s} | {'Entropy':8s} | {'Test PPL':8s} | {'PPL Gap %':9s} | {'2B NPZ (B)':12s} | {'Gzip Planes':12s} | {'RAM 2B/R3':13s} | {'Decode Tok/s':12s}")
    print("-" * 135)
    
    ref_ppl = 6.3703
    for r in results:
        gap = (r['test_ppl'] - ref_ppl) / ref_ppl * 100.0
        print(f"{r['candidate_id']:10s} | {r['type']:7s} | {r['param']:7.3f} | {r['zero_pct']:7.2f}% | {r['entropy']:7.4f} | {r['test_ppl']:8.4f} | {gap:+8.2f}% | {r['size_2b_npz_bytes']:12,d} | {r['size_gzip_planes_bytes']:12,d} | 681K / 570K   | {r['throughput_2b']:10.1f} 2B / {r['throughput_r3']:6.1f} R3")
    print("=" * 135)

    print(f"\nTotal Experiment Execution Time: {time.perf_counter() - t_exp_start:.2f} s")


if __name__ == '__main__':
    run_phase3_6_experiment()
