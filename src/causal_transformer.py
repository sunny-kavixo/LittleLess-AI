"""
1.17M Parameter Causal Transformer for Apple Silicon M4 Research
Features:
- Causal Multi-Head Self-Attention (Pre-LN)
- LayerNorm, FeedForward (GELU)
- Flexible per-module quantization hooks (FP16, Ternary, Dual-Basis, Sparse Residual, LoRA)
- Autoregressive generation
- Byte-level tokenization (vocab=256, zero OOV)
- Context length T=128
"""

import math, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Quantization Functions with Straight-Through Estimators (STE)
def quant_ternary_global(w):
    threshold = 0.7 * w.detach().abs().mean()
    mask = w.detach().abs() > threshold
    scale = w.detach().abs()[mask].mean() if bool(mask.any()) else w.new_tensor(1.0)
    q = scale * w.detach().sign() * mask
    return w + (q - w).detach()

def quant_ternary_per_channel(w):
    threshold = 0.7 * w.detach().abs().mean(dim=-1, keepdim=True)
    mask = w.detach().abs() > threshold
    scale = (w.detach().abs() * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    q = scale * w.detach().sign() * mask
    return w + (q - w).detach()

def quant_dual_basis_global(w):
    # Stage 1: Coarse ternary
    t1 = 0.7 * w.detach().abs().mean()
    m1 = w.detach().abs() > t1
    s1 = w.detach().abs()[m1].mean() if bool(m1.any()) else w.new_tensor(1.0)
    q1 = s1 * w.detach().sign() * m1
    
    # Stage 2: Fine residual ternary
    res = w.detach() - q1
    t2 = 0.7 * res.abs().mean()
    m2 = res.abs() > t2
    s2 = res.abs()[m2].mean() if bool(m2.any()) else w.new_tensor(1.0)
    q2 = s2 * res.sign() * m2
    
    q = q1 + q2
    return w + (q - w).detach()

def quant_dual_basis_per_channel(w):
    # Stage 1: Per-channel ternary
    t1 = 0.7 * w.detach().abs().mean(dim=-1, keepdim=True)
    m1 = w.detach().abs() > t1
    s1 = (w.detach().abs() * m1).sum(dim=-1, keepdim=True) / m1.sum(dim=-1, keepdim=True).clamp(min=1.0)
    q1 = s1 * w.detach().sign() * m1
    
    # Stage 2: Per-channel residual ternary
    res = w.detach() - q1
    t2 = 0.7 * res.abs().mean(dim=-1, keepdim=True)
    m2 = res.abs() > t2
    s2 = (res.abs() * m2).sum(dim=-1, keepdim=True) / m2.sum(dim=-1, keepdim=True).clamp(min=1.0)
    q2 = s2 * res.sign() * m2
    
    q = q1 + q2
    return w + (q - w).detach()

def quant_sparse_residual(w, top_pct=0.03):
    t1 = 0.7 * w.detach().abs().mean()
    m1 = w.detach().abs() > t1
    s1 = w.detach().abs()[m1].mean() if bool(m1.any()) else w.new_tensor(1.0)
    q1 = s1 * w.detach().sign() * m1
    
    res = w.detach() - q1
    k = max(1, int(w.numel() * top_pct))
    cutoff = torch.topk(res.abs().view(-1), k).values[-1]
    res_sparse = torch.where(res.abs() >= cutoff, res, torch.zeros_like(res))
    q = q1 + res_sparse
    return w + (q - w).detach()

def quant_low_rank(w, rank=4):
    t1 = 0.7 * w.detach().abs().mean()
    m1 = w.detach().abs() > t1
    s1 = w.detach().abs()[m1].mean() if bool(m1.any()) else w.new_tensor(1.0)
    q1 = s1 * w.detach().sign() * m1
    
    res = w.detach() - q1
    # Bounded SVD
    U, S, Vh = torch.linalg.svd(res, full_matrices=False)
    r = min(rank, len(S))
    res_lora = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
    q = q1 + res_lora
    return w + (q - w).detach()

def apply_weight_quant(w, mode):
    if mode == 'fp16':
        return w
    elif mode == 'ternary_global':
        return quant_ternary_global(w)
    elif mode == 'ternary_pc':
        return quant_ternary_per_channel(w)
    elif mode == 'dbq_global':
        return quant_dual_basis_global(w)
    elif mode == 'dbq_pc':
        return quant_dual_basis_per_channel(w)
    elif mode == 'sparse_res_3pct':
        return quant_sparse_residual(w, top_pct=0.03)
    elif mode == 'lowrank_r4':
        return quant_low_rank(w, rank=4)
    elif mode == 'degraded_control':
        # Deliberately degraded negative control (1-bit extreme thresholding + random sign)
        m = w.detach().abs() > w.detach().abs().median()
        return (w.detach().sign() * m).float()
    else:
        raise ValueError(f"Unknown quant mode: {mode}")

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x, quant_modes):
        B, T, C = x.size()
        
        wq = apply_weight_quant(self.q_proj.weight, quant_modes.get('attn_q', 'fp16'))
        wk = apply_weight_quant(self.k_proj.weight, quant_modes.get('attn_k', 'fp16'))
        wv = apply_weight_quant(self.v_proj.weight, quant_modes.get('attn_v', 'fp16'))
        wo = apply_weight_quant(self.out_proj.weight, quant_modes.get('attn_out', 'fp16'))
        
        q = F.linear(x, wq).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = F.linear(x, wk).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = F.linear(x, wv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v # [B, n_heads, T, d_head]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return F.linear(y, wo)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_fc1 = nn.Linear(d_model, d_ff)
        self.mlp_fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x, quant_modes):
        x = x + self.attn(self.ln1(x), quant_modes)
        
        w_fc1 = apply_weight_quant(self.mlp_fc1.weight, quant_modes.get('mlp_fc1', 'fp16'))
        w_fc2 = apply_weight_quant(self.mlp_fc2.weight, quant_modes.get('mlp_fc2', 'fp16'))
        
        h = self.ln2(x)
        h = F.gelu(F.linear(h, w_fc1, self.mlp_fc1.bias))
        h = F.linear(h, w_fc2, self.mlp_fc2.bias)
        x = x + h
        return x

class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=160, n_heads=5, n_layers=4, d_ff=512, block_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, block_size) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.quant_modes = {} # default fp16

    def set_quant_modes(self, modes):
        self.quant_modes = modes

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        w_emb = apply_weight_quant(self.wte.weight, self.quant_modes.get('emb', 'fp16'))
        w_pos = apply_weight_quant(self.wpe.weight, self.quant_modes.get('pos', 'fp16'))
        
        tok_emb = F.embedding(idx, w_emb)
        pos_emb = F.embedding(pos, w_pos)
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x, self.quant_modes)
            
        x = self.ln_f(x)
        w_head = apply_weight_quant(self.lm_head.weight, self.quant_modes.get('head', 'fp16'))
        logits = F.linear(x, w_head)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_tok), dim=1)
        return idx
