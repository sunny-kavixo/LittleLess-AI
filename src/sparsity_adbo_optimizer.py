"""
Sparsity-Aware Alternating Dual-Basis Optimization (S-ADBO)
LittleLess AI — Phase 3.6 Controlled Experiment

Investigates whether sparsity-aware ternary optimization with curvature-normalized
L0 penalty reduces serialized artifact size and active packed-weight memory while
preserving language-model perplexity and native inference throughput.

Formulation:
  min_{T1, T2, s1, s2} Tr((W - W_hat) H (W - W_hat)^T) + lambda * sum_j H_jj * (1_{T1 != 0} + 1_{T2 != 0})
"""

import copy, json, math, os, sys, time
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

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 9 candidate coordinate pairs
PAIRS = torch.tensor([
    [-1, -1], [-1, 0], [-1, 1],
    [ 0, -1], [ 0, 0], [ 0, 1],
    [ 1, -1], [ 1, 0], [ 1, 1]
], dtype=torch.float32)
CAND_T1 = PAIRS[:, 0].view(1, 9)
CAND_T2 = PAIRS[:, 1].view(1, 9)

# Non-zero trit count per pair: [2, 1, 2, 1, 0, 1, 2, 1, 2]
PAIR_NONZEROS = (PAIRS != 0).sum(dim=1).float().view(1, 9)


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


def optimize_layer_sadbo(W, H, c_thresh=0.7, sweeps=2, sparsity_lambda=0.0):
    """
    Performs Sparsity-Aware Alternating Dual-Basis Optimization (S-ADBO) on weight matrix W.
    W: [M, N]
    H: [N, N]
    sparsity_lambda: penalty coefficient for non-zero trits, scaled by H_jj
    """
    M, N = W.shape
    
    # 1. Greedy AWG-DBQ initialization
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
    
    # Precompute non-zero penalty vector [1, 9] on device
    pair_nonzeros = PAIR_NONZEROS.to(W.device)
    cand_t1 = CAND_T1.to(W.device)
    cand_t2 = CAND_T2.to(W.device)
    
    # 2. Alternating sweeps: Vectorized Discrete S-ADBO Coordinate Descent
    for s_idx in range(sweeps):
        t0 = time.perf_counter()
        G = R @ H
        flips = 0
        
        for j in range(N):
            H_jj = H_diag[j]
            curr_t1 = T1[:, j:j+1]
            curr_t2 = T2[:, j:j+1]
            
            delta = s1 * (cand_t1 - curr_t1) + s2 * (cand_t2 - curr_t2) # [M, 9]
            d_mse = -2.0 * delta * G[:, j:j+1] + (delta ** 2) * H_jj     # [M, 9]
            
            # Curvature-normalized L0 sparsity penalty: lambda * H_jj * count(nonzeros)
            if sparsity_lambda > 0.0:
                d_reg = sparsity_lambda * H_jj * pair_nonzeros # [1, 9]
                total_cost = d_mse + d_reg
            else:
                total_cost = d_mse
            
            best_idx = torch.argmin(total_cost, dim=1) # [M]
            best_t1 = cand_t1[0, best_idx]             # [M]
            best_t2 = cand_t2[0, best_idx]             # [M]
            
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


def naive_magnitude_prune_layer(T1, s1, T2, s2, W_ref, thresh_factor=0.8):
    """
    Naive baseline: zeroes out trits where the reconstructed weight magnitude |W_hat|
    is below thresh_factor * std(W_ref). Then refits scales s1, s2.
    """
    if isinstance(s1, np.ndarray):
        s1 = torch.from_numpy(s1.astype(np.float32))
    if isinstance(s2, np.ndarray):
        s2 = torch.from_numpy(s2.astype(np.float32))
    if hasattr(s1, 'to'): s1 = s1.to(W_ref.device)
    if hasattr(s2, 'to'): s2 = s2.to(W_ref.device)
    if hasattr(T1, 'to'): T1 = T1.to(W_ref.device)
    if hasattr(T2, 'to'): T2 = T2.to(W_ref.device)
    if s1.ndim == 1: s1 = s1.unsqueeze(1)
    if s2.ndim == 1: s2 = s2.unsqueeze(1)
    
    W_hat = s1 * T1 + s2 * T2
    threshold = thresh_factor * W_ref.std()
    prune_mask = W_hat.abs() < threshold
    
    T1_pruned = T1.clone()
    T2_pruned = T2.clone()
    T1_pruned[prune_mask] = 0.0
    T2_pruned[prune_mask] = 0.0
    
    # Refit scales with simple least squares
    s1_new = (W_ref * T1_pruned).sum(dim=-1, keepdim=True) / (T1_pruned**2).sum(dim=-1, keepdim=True).clamp(min=1.0)
    R1 = W_ref - s1_new * T1_pruned
    s2_new = (R1 * T2_pruned).sum(dim=-1, keepdim=True) / (T2_pruned**2).sum(dim=-1, keepdim=True).clamp(min=1.0)
    
    W_hat_pruned = s1_new * T1_pruned + s2_new * T2_pruned
    return T1_pruned, s1_new, T2_pruned, s2_new, W_hat_pruned


def compute_trit_statistics(t1_list, t2_list):
    """
    Computes zero percentage, symbol frequencies, and empirical Shannon entropy.
    """
    all_trits = []
    for t1, t2 in zip(t1_list, t2_list):
        if hasattr(t1, 'cpu'): t1 = t1.cpu().numpy()
        if hasattr(t2, 'cpu'): t2 = t2.cpu().numpy()
        all_trits.append(t1.ravel())
        all_trits.append(t2.ravel())
    all_trits = np.concatenate(all_trits)
    
    total_count = len(all_trits)
    count_neg = int(np.sum(all_trits == -1))
    count_zero = int(np.sum(all_trits == 0))
    count_pos = int(np.sum(all_trits == 1))
    
    p_neg = count_neg / total_count
    p_zero = count_zero / total_count
    p_pos = count_pos / total_count
    
    # Shannon entropy
    entropy = 0.0
    for p in [p_neg, p_zero, p_pos]:
        if p > 0:
            entropy -= p * math.log2(p)
            
    theoretical_ternary_bytes = (total_count * entropy) / 8.0
    
    return {
        'total_trits': total_count,
        'count_neg': count_neg,
        'count_zero': count_zero,
        'count_pos': count_pos,
        'p_neg': p_neg,
        'p_zero': p_zero,
        'p_pos': p_pos,
        'zero_pct': p_zero * 100.0,
        'entropy_bits_per_trit': entropy,
        'theoretical_ternary_bytes': theoretical_ternary_bytes
    }
