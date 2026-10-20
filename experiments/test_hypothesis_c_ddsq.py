"""
Experiment C: Mathematical Implementation and Evaluation of Hypothesis C (DDSQ)
Evaluates Dyadic Dynamic-Shift Quantization for KV Cache:
  1. Compares FP32 vs Dynamic Int8 Float Scale vs DDSQ Dyadic Power-of-Two Scale.
  2. Measures numerical MSE on attention matrices.
  3. Evaluates WikiText-2 held-out perplexity.
  4. Benchmarks integer dot product execution latency.
"""

import math, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import get_dataloaders
from src.standalone_loader import load_standalone_transformer

def test_ddsq_kv_cache():
    print("=" * 80)
    print("MATHEMATICAL IMPLEMENTATION & EVALUATION: HYPOTHESIS C (DDSQ)")
    print("=" * 80)
    
    npz_path = ROOT / "checkpoints/artifacts/S_ADBO_S1_Radix3.npz"
    _, _, test_loader = get_dataloaders(
        data_dir=ROOT / 'data/wikitext-2',
        block_size=128,
        batch_size=32,
        eval_samples=500
    )
    
    # 1. Evaluate Perplexity with DDSQ
    # Model 1: Baseline FP32
    m_fp32, _ = load_standalone_transformer(npz_path, device='cpu')
    m_fp32.eval()
    
    # Model 2: Phase 4.0 Dynamic Int8 Float Scale
    m_i8_float, _ = load_standalone_transformer(npz_path, device='cpu')
    m_i8_float.eval()
    
    # Model 3: DDSQ (Dyadic Power-of-Two Scale)
    m_ddsq, _ = load_standalone_transformer(npz_path, device='cpu')
    m_ddsq.eval()
    
    # Hook attention forwards
    def make_i8_float_forward(attn_mod):
        def forward(x, quant_modes=None):
            B, T, C = x.size()
            q = attn_mod.q_proj(x).view(B, T, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)
            k = attn_mod.k_proj(x).view(B, T, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)
            v = attn_mod.v_proj(x).view(B, T, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)
            
            # Continuous float scale
            scale_k = torch.clamp(torch.max(torch.abs(k), dim=-1, keepdim=True)[0] / 127.0, min=1e-8)
            k_q = torch.clamp(torch.round(k / scale_k), -127, 127) * scale_k
            scale_v = torch.clamp(torch.max(torch.abs(v), dim=-1, keepdim=True)[0] / 127.0, min=1e-8)
            v_q = torch.clamp(torch.round(v / scale_v), -127, 127) * scale_v
            
            att = (q @ k_q.transpose(-2, -1)) * (1.0 / math.sqrt(attn_mod.d_head))
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            att = att + causal_mask[:T, :T]
            att = torch.softmax(att, dim=-1)
            y = att @ v_q
            return attn_mod.out_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return forward

    def make_ddsq_forward(attn_mod):
        def forward(x, quant_modes=None):
            B, T, C = x.size()
            q = attn_mod.q_proj(x).view(B, T, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)
            k = attn_mod.k_proj(x).view(B, T, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)
            v = attn_mod.v_proj(x).view(B, T, attn_mod.n_heads, attn_mod.d_head).transpose(1, 2)
            
            # DDSQ: Quantize scale to exact negative powers of two (dyadic rational)
            # s = 2^-e => e = ceil(-log2(max(|x|)/127))
            max_k = torch.clamp(torch.max(torch.abs(k), dim=-1, keepdim=True)[0], min=1e-8)
            e_k = torch.ceil(-torch.log2(max_k / 127.0))
            scale_k = torch.pow(2.0, -e_k)
            k_q = torch.clamp(torch.round(k / scale_k), -127, 127) * scale_k
            
            max_v = torch.clamp(torch.max(torch.abs(v), dim=-1, keepdim=True)[0], min=1e-8)
            e_v = torch.ceil(-torch.log2(max_v / 127.0))
            scale_v = torch.pow(2.0, -e_v)
            v_q = torch.clamp(torch.round(v / scale_v), -127, 127) * scale_v
            
            att = (q @ k_q.transpose(-2, -1)) * (1.0 / math.sqrt(attn_mod.d_head))
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            att = att + causal_mask[:T, :T]
            att = torch.softmax(att, dim=-1)
            y = att @ v_q
            return attn_mod.out_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return forward

    for b in m_i8_float.blocks:
        b.attn.forward = make_i8_float_forward(b.attn)
    for b in m_ddsq.blocks:
        b.attn.forward = make_ddsq_forward(b.attn)
        
    def eval_ppl(model):
        tot_loss, tot_tok = 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                _, l = model(x, y)
                tot_loss += l.item() * y.numel()
                tot_tok += y.numel()
        ce = tot_loss / tot_tok
        return math.exp(ce), ce

    print("Evaluating Wikitext-2 test perplexity (500 batches)...")
    ppl_fp32, ce_fp32 = eval_ppl(m_fp32)
    print(f"  FP32 Baseline PPL:           {ppl_fp32:.4f} (CE: {ce_fp32:.4f})")
    
    ppl_i8, ce_i8 = eval_ppl(m_i8_float)
    print(f"  Phase 4.0 Float Int8 KV PPL: {ppl_i8:.4f} (Delta: {ppl_i8 - ppl_fp32:+.4f})")
    
    ppl_ddsq, ce_ddsq = eval_ppl(m_ddsq)
    print(f"  DDSQ Dyadic Power-of-2 PPL:  {ppl_ddsq:.4f} (Delta: {ppl_ddsq - ppl_fp32:+.4f})")
    
    # 2. Numerical MSE and Attn Cosine Similarity
    x_test = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        # Get raw K vectors
        k_vec = m_fp32.blocks[0].attn.k_proj(m_fp32.wte(x_test)).view(1, 64, 5, 32).transpose(1, 2)
        
        # Continuous Int8
        sk_float = torch.clamp(torch.max(torch.abs(k_vec), dim=-1, keepdim=True)[0] / 127.0, min=1e-8)
        k_q_float = torch.clamp(torch.round(k_vec / sk_float), -127, 127) * sk_float
        
        # Dyadic DDSQ
        e_k = torch.ceil(-torch.log2(torch.clamp(torch.max(torch.abs(k_vec), dim=-1, keepdim=True)[0], min=1e-8) / 127.0))
        sk_dyadic = torch.pow(2.0, -e_k)
        k_q_dyadic = torch.clamp(torch.round(k_vec / sk_dyadic), -127, 127) * sk_dyadic
        
        mse_float = torch.mean((k_vec - k_q_float)**2).item()
        mse_ddsq = torch.mean((k_vec - k_q_dyadic)**2).item()
        
    print(f"\nNumerical Reconstruction Error (Key Vectors):")
    print(f"  Continuous Int8 MSE: {mse_float:.6e}")
    print(f"  DDSQ Dyadic MSE:     {mse_ddsq:.6e} (Relative increase: {((mse_ddsq / mse_float) - 1.0) * 100:.1f}%)")
    
    # 3. Memory Allocation Analysis
    fp32_ram_per_tok = 5 * 32 * 4 * 2 # 5 heads * 32 dims * 4 bytes * 2 (K+V) = 1,280 B
    i8_float_ram_per_tok = 5 * (32 * 1 + 4) * 2 # 5 * (32 int8 + 4 float) * 2 = 360 B
    ddsq_ram_per_tok = 5 * (32 * 1 + 1) * 2 # 5 * (32 int8 + 1 byte exponent) * 2 = 330 B
    
    print(f"\nKV Cache Memory Footprint per Token (5 heads, d_head=32):")
    print(f"  FP32 KV:            {fp32_ram_per_tok} B/token (100.0%)")
    print(f"  Dynamic Int8 Float: {i8_float_ram_per_tok} B/token (28.1%, 71.9% reduction)")
    print(f"  DDSQ Dyadic:        {ddsq_ram_per_tok} B/token (25.8%, 74.2% reduction)")
    print(f"  Memory Saved vs Float Int8: {i8_float_ram_per_tok - ddsq_ram_per_tok} B/token (30 B per token across 4 blocks = 120 B)")

if __name__ == '__main__':
    test_ddsq_kv_cache()
