"""
Activation-Weighted Gram Dual-Basis Quantization (AWG-DBQ)
LittleLess AI — Phase 3.1 Mathematical Invention

Mathematical Derivation:
Approximates weight matrix W in R^{M x N} with two ternary planes:
  W_hat = s1 * T1 + s2 * T2
where T1, T2 in {-1, 0, +1}^{M x N} and s1, s2 in R^M.

To minimize the true layer activation reconstruction error:
  L = || W X - W_hat X ||_F^2 = Tr( (W - W_hat) H (W - W_hat)^T )
where H = (X X^T) / N is the empirical input covariance matrix.

For each row i:
  [ s1_i ]   [ T1_i H T1_i^T   T1_i H T2_i^T ]^{-1}  [ T1_i H W_i^T ]
  [ s2_i ] = [ T2_i H T1_i^T   T2_i H T2_i^T ]       [ T2_i H W_i^T ]
"""

import math, numpy as np, torch

def quantize_awg_dbq_matrix(W, H):
    """
    W: torch.Tensor [M, N]
    H: torch.Tensor [N, N] (input covariance)
    Returns:
      T1: torch.Tensor [M, N] in {-1, 0, 1}
      s1: torch.Tensor [M, 1]
      T2: torch.Tensor [M, N] in {-1, 0, 1}
      s2: torch.Tensor [M, 1]
      W_hat: torch.Tensor [M, N]
    """
    M, N = W.shape
    
    # Plane 1: Initial ternary pattern
    t1 = 0.7 * W.abs().mean(dim=-1, keepdim=True)
    m1 = W.abs() > t1
    T1 = W.sign() * m1
    
    denom1 = (T1**2).sum(dim=-1, keepdim=True).clamp(min=1.0)
    s1_init = (W * T1).sum(dim=-1, keepdim=True) / denom1
    R1 = W - s1_init * T1
    
    # Plane 2: Residual ternary pattern
    t2 = 0.7 * R1.abs().mean(dim=-1, keepdim=True)
    m2 = R1.abs() > t2
    T2 = R1.sign() * m2
    
    # Activation-Weighted Gram Normal Equations
    T1_H = T1 @ H
    T2_H = T2 @ H
    W_H = W @ H
    
    g11 = (T1_H * T1).sum(dim=-1, keepdim=True)
    g12 = (T1_H * T2).sum(dim=-1, keepdim=True)
    g22 = (T2_H * T2).sum(dim=-1, keepdim=True)
    
    b1 = (W_H * T1).sum(dim=-1, keepdim=True)
    b2 = (W_H * T2).sum(dim=-1, keepdim=True)
    
    det = (g11 * g22 - g12**2).clamp(min=1e-8)
    s1 = (g22 * b1 - g12 * b2) / det
    s2 = (g11 * b2 - g12 * b1) / det
    
    W_hat = s1 * T1 + s2 * T2
    return T1, s1, T2, s2, W_hat

def pack_ternary_planes(T1, s1, T2, s2):
    """
    Packs two ternary planes into 2-bit planes with float16 scales.
    Each trit in {-1, 0, 1} mapped to {0, 1, 2} encoded in 2 bits.
    """
    # Plane 1
    c1 = T1.cpu().numpy().astype(np.int8).ravel() + 1
    bits1 = np.stack([(c1 >> 1) & 1, c1 & 1], 1).ravel()
    p1 = np.packbits(bits1, bitorder='little')
    
    # Plane 2
    c2 = T2.cpu().numpy().astype(np.int8).ravel() + 1
    bits2 = np.stack([(c2 >> 1) & 1, c2 & 1], 1).ravel()
    p2 = np.packbits(bits2, bitorder='little')
    
    return {
        'p1': p1,
        's1': s1.cpu().numpy().astype(np.float16),
        'p2': p2,
        's2': s2.cpu().numpy().astype(np.float16)
    }

def unpack_ternary_planes(p1, s1, p2, s2, shape):
    """
    Unpacks 2-bit planes and scales back to float32 matrix.
    """
    rows, cols = shape
    b1 = np.unpackbits(p1, bitorder='little')[:rows * cols * 2].reshape(-1, 2)
    c1 = (b1[:, 0].astype(np.int8) << 1 | b1[:, 1].astype(np.int8)) - 1
    t1 = c1.reshape(rows, cols).astype(np.float32)
    
    b2 = np.unpackbits(p2, bitorder='little')[:rows * cols * 2].reshape(-1, 2)
    c2 = (b2[:, 0].astype(np.int8) << 1 | b2[:, 1].astype(np.int8)) - 1
    t2 = c2.reshape(rows, cols).astype(np.float32)
    
    return (s1.astype(np.float32) * t1 + s2.astype(np.float32) * t2)

def quantize_int8_embedding(w):
    scale = float(w.abs().max() / 127.0)
    arr = w.cpu().numpy()
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, np.array([scale], dtype=np.float16)
