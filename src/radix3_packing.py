"""
Radix-3 (Base-3) 5-Trit Lossless Packing for Ternary Neural Representations
LittleLess AI — Phase 3.5 Core Mathematical Invention

Row-Aligned Architecture:
Packs each row of a matrix independently into (N + 4) // 5 uint8 bytes.
Guarantees O(1) row pointer arithmetic: row_p = p + i * bytes_per_row.
"""

import numpy as np

def pack_radix3_matrix(T: np.ndarray):
    """
    Packs a 2D matrix [M, N] of trits in {-1, 0, 1} into row-aligned Radix-3 uint8 bytes.
    Each row is padded to a multiple of 5 trits.
    Returns:
      packed: 1D np.ndarray (uint8) of length M * ((N + 4) // 5)
    """
    M, N = T.shape
    bytes_per_row = (N + 4) // 5
    
    pad = (5 - (N % 5)) % 5
    if pad > 0:
        T_padded = np.pad(T, ((0, 0), (0, pad)), constant_values=0)
    else:
        T_padded = T
        
    c = (T_padded + 1).astype(np.uint8) # [M, bytes_per_row * 5]
    c_groups = c.reshape(M, bytes_per_row, 5)
    
    # Horner's polynomial evaluation
    b = c_groups[:, :, 0] + 3 * (
        c_groups[:, :, 1] + 3 * (
            c_groups[:, :, 2] + 3 * (
                c_groups[:, :, 3] + 3 * c_groups[:, :, 4]
            )
        )
    )
    return b.ravel()

# Precomputed fast decode lookup table: 256 -> 5 trits in {-1, 0, 1}
RADIX3_DECODE_LUT = np.zeros((256, 5), dtype=np.int8)
for b in range(256):
    temp = b
    c0 = temp % 3; temp //= 3
    c1 = temp % 3; temp //= 3
    c2 = temp % 3; temp //= 3
    c3 = temp % 3; temp //= 3
    c4 = temp % 3
    RADIX3_DECODE_LUT[b] = [c0 - 1, c1 - 1, c2 - 1, c3 - 1, c4 - 1]

def unpack_radix3_matrix(packed_bytes: np.ndarray, M: int, N: int):
    """
    Unpacks row-aligned Radix-3 packed uint8 bytes back to [M, N] trits in {-1, 0, 1}.
    """
    bytes_per_row = (N + 4) // 5
    b_2d = packed_bytes.reshape(M, bytes_per_row)
    trits_3d = RADIX3_DECODE_LUT[b_2d] # [M, bytes_per_row, 5]
    trits_2d = trits_3d.reshape(M, bytes_per_row * 5)[:, :N]
    return trits_2d

def pack_radix3_trits(arr: np.ndarray):
    """
    Packs arbitrary 1D/ND array of trits in {-1, 0, 1} into Radix-3 uint8 bytes.
    Returns (packed_bytes, orig_shape, total_elements).
    """
    orig_shape = arr.shape
    n = arr.size
    flat = arr.ravel()
    pad = (5 - (n % 5)) % 5
    if pad > 0:
        flat = np.pad(flat, (0, pad), constant_values=0)
    c = (flat + 1).astype(np.uint8).reshape(-1, 5)
    b = c[:, 0] + 3 * (c[:, 1] + 3 * (c[:, 2] + 3 * (c[:, 3] + 3 * c[:, 4])))
    return b.astype(np.uint8), orig_shape, n

def unpack_radix3_bytes(packed: np.ndarray, orig_shape, n: int):
    """
    Unpacks Radix-3 uint8 bytes back to original trit array shape.
    """
    trits = RADIX3_DECODE_LUT[packed].ravel()[:n]
    return trits.reshape(orig_shape)

def pack_radix3_layer(T1, s1, T2, s2):
    """
    Packs two ternary planes T1, T2 using row-aligned Radix-3 5-trit packing.
    """
    if hasattr(T1, 'cpu'): T1 = T1.cpu().numpy()
    if hasattr(T2, 'cpu'): T2 = T2.cpu().numpy()
    if hasattr(s1, 'cpu'): s1 = s1.cpu().numpy()
    if hasattr(s2, 'cpu'): s2 = s2.cpu().numpy()
    
    M, N = T1.shape
    p1_bytes = pack_radix3_matrix(T1)
    p2_bytes = pack_radix3_matrix(T2)
    
    return {
        'p1_radix3': p1_bytes,
        's1': s1.astype(np.float16),
        'p2_radix3': p2_bytes,
        's2': s2.astype(np.float16),
        'shape': np.array([M, N], dtype=np.int32)
    }

def unpack_radix3_layer(p1_bytes, s1, p2_bytes, s2, shape):
    """
    Unpacks Radix-3 layer back to float32 matrix W_hat = s1*T1 + s2*T2.
    """
    M, N = shape
    T1 = unpack_radix3_matrix(p1_bytes, M, N).astype(np.float32)
    T2 = unpack_radix3_matrix(p2_bytes, M, N).astype(np.float32)
    
    s1_f = s1.astype(np.float32).reshape(-1, 1)
    s2_f = s2.astype(np.float32).reshape(-1, 1)
    
    return s1_f * T1 + s2_f * T2
