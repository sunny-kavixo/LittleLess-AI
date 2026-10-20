"""
Standalone Causal Transformer Loader & Inference Engine
Loads compressed LittleLess artifacts (.npz + .json) and reconstructs active inference model.
Provides standalone execution independent of training checkpoints.
"""

from pathlib import Path
import json, numpy as np, torch
from src.causal_transformer import CausalTransformerLM

def load_standalone_transformer(artifact_path, device='mps', target_dtype=None):
    if target_dtype is None:
        target_dtype = torch.float16 if device == 'mps' else torch.float32
    artifact_path = Path(artifact_path)
    if artifact_path.suffix == '.json':
        npz_path = artifact_path.with_suffix('.npz')
        json_path = artifact_path
    else:
        npz_path = artifact_path
        json_path = artifact_path.with_suffix('.json')
        
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    npz = np.load(npz_path)
    
    model = CausalTransformerLM().to(device=device, dtype=target_dtype)
    state = model.state_dict()
    
    for name in list(state.keys()):
        if f'{name}.int8' in npz:
            q = npz[f'{name}.int8']
            s = npz[f'{name}.scale']
            W = (q.astype(np.float32) * s.astype(np.float32)).astype(np.float32)
            state[name] = torch.from_numpy(W).to(device=device, dtype=target_dtype)
        elif f'{name}.p1' in npz:
            p1 = npz[f'{name}.p1']
            s1 = npz[f'{name}.s1']
            p2 = npz[f'{name}.p2']
            s2 = npz[f'{name}.s2']
            
            rows = s1.shape[0]
            cols = state[name].shape[1]
            
            b1 = np.unpackbits(p1, bitorder='little')[:rows * cols * 2].reshape(-1, 2)
            c1 = (b1[:, 0].astype(np.int8) << 1 | b1[:, 1].astype(np.int8)) - 1
            t1 = c1.reshape(rows, cols).astype(np.float32)
            
            b2 = np.unpackbits(p2, bitorder='little')[:rows * cols * 2].reshape(-1, 2)
            c2 = (b2[:, 0].astype(np.int8) << 1 | b2[:, 1].astype(np.int8)) - 1
            t2 = c2.reshape(rows, cols).astype(np.float32)
            
            W = (s1.astype(np.float32) * t1 + s2.astype(np.float32) * t2)
            state[name] = torch.from_numpy(W).to(device=device, dtype=target_dtype)
        elif f'{name}.p1_radix3' in npz:
            from src.radix3_packing import unpack_radix3_matrix
            p1 = npz[f'{name}.p1_radix3']
            s1 = npz[f'{name}.s1']
            p2 = npz[f'{name}.p2_radix3']
            s2 = npz[f'{name}.s2']
            rows = s1.shape[0]
            cols = state[name].shape[1]
            t1 = unpack_radix3_matrix(p1, rows, cols).astype(np.float32)
            t2 = unpack_radix3_matrix(p2, rows, cols).astype(np.float32)
            W = (s1.astype(np.float32).reshape(-1, 1) * t1 + s2.astype(np.float32).reshape(-1, 1) * t2)
            state[name] = torch.from_numpy(W).to(device=device, dtype=target_dtype)
        elif f'{name}.val' in npz:
            state[name] = torch.from_numpy(npz[f'{name}.val'].astype(np.float32)).to(device=device, dtype=target_dtype)
            
    model.load_state_dict(state)
    model.eval()
    return model, meta
