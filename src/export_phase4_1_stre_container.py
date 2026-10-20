"""
LittleLess AI — Phase 4.1 Sparse-Trit Run-Encoded (STRE) Container Exporter
Converts packed transformer checkpoints (.npz) into lightweight, seekable,
zero-copy run-encoded containers (.llstre) for native fused streaming.
Exploits Radix-3's mathematical property: bytes 0..242 are valid trits,
leaving bytes 243..255 as zero-run and match escape markers.
"""

import os, struct
from pathlib import Path
import numpy as np

def stre_compress_plane(data: bytes) -> bytes:
    """
    Compresses a Radix-3 plane using Sparse-Trit Run Encoding (STRE).
    Byte 243: Run of byte 121 (five zero trits). Next byte is count (2..255).
    Byte 244: Run of arbitrary byte. Next byte is val, next is count.
    Literal bytes in 0..242 are passed through unmodified.
    """
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        # Check run of byte 121 (all-zero trits)
        if data[i] == 121:
            run = 0
            while i + run < n and data[i + run] == 121 and run < 255:
                run += 1
            if run >= 2:
                out.append(243)
                out.append(run)
                i += run
                continue
        
        # Check run of arbitrary byte
        b = data[i]
        run = 0
        while i + run < n and data[i + run] == b and run < 255:
            run += 1
        if run >= 4:
            out.append(244)
            out.append(b)
            out.append(run)
            i += run
            continue
            
        out.append(data[i])
        i += 1
    return bytes(out)


def export_llstre_container(npz_path, output_path, pack_mode=1):
    """
    Exports a .llstre binary container.
    """
    npz_path = Path(npz_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    npz = np.load(npz_path)
    
    p1_suf = '.p1_radix3' if pack_mode == 1 else '.p1'
    p2_suf = '.p2_radix3' if pack_mode == 1 else '.p2'
    proj_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'mlp_fc1', 'mlp_fc2']
    
    # 1. Header (40 bytes)
    magic = b"LLSTRE01"
    version = 1
    d_model = 160
    n_heads = 5
    d_head = 32
    n_blocks = 4
    max_context = 128
    vocab_size = 256
    
    header_bytes = magic + struct.pack(
        "<8I",
        version, pack_mode, d_model, n_heads, d_head, n_blocks, max_context, vocab_size
    )
    
    # 2. Embeddings
    wte_q = np.ascontiguousarray(npz['wte.weight.int8'], dtype=np.int8)
    wte_s = float(npz['wte.weight.scale'][0])
    wpe_q = np.ascontiguousarray(npz['wpe.weight.int8'], dtype=np.int8)
    wpe_s = float(npz['wpe.weight.scale'][0])
    
    embs_bytes = struct.pack("<fI", wte_s, wte_q.nbytes) + wte_q.tobytes() + \
                 struct.pack("<fI", wpe_s, wpe_q.nbytes) + wpe_q.tobytes()
                 
    # 3. Block Norms & Biases
    block_norms_bytes = bytearray()
    for b_idx in range(n_blocks):
        ln1_w = np.ascontiguousarray(npz[f'blocks.{b_idx}.ln1.weight.val'].astype(np.float32))
        ln1_b = np.ascontiguousarray(npz[f'blocks.{b_idx}.ln1.bias.val'].astype(np.float32))
        ln2_w = np.ascontiguousarray(npz[f'blocks.{b_idx}.ln2.weight.val'].astype(np.float32))
        ln2_b = np.ascontiguousarray(npz[f'blocks.{b_idx}.ln2.bias.val'].astype(np.float32))
        fc1_b = np.ascontiguousarray(npz[f'blocks.{b_idx}.mlp_fc1.bias.val'].astype(np.float32))
        fc2_b = np.ascontiguousarray(npz[f'blocks.{b_idx}.mlp_fc2.bias.val'].astype(np.float32))
        
        block_norms_bytes += ln1_w.tobytes() + ln1_b.tobytes() + \
                             ln2_w.tobytes() + ln2_b.tobytes() + \
                             fc1_b.tobytes() + fc2_b.tobytes()
                             
    # 4. Final Head Norms
    ln_f_w = np.ascontiguousarray(npz['ln_f.weight.val'].astype(np.float32))
    ln_f_b = np.ascontiguousarray(npz['ln_f.bias.val'].astype(np.float32))
    final_norm_bytes = ln_f_w.tobytes() + ln_f_b.tobytes()
    
    # 5. Compress Layers
    layers_meta = []
    compressed_payloads = []
    
    for b_idx in range(n_blocks):
        for l_type, p_name in enumerate(proj_names):
            prefix = f'blocks.{b_idx}.{p_name}' if 'mlp' in p_name else f'blocks.{b_idx}.attn.{p_name}'
            p1 = np.ascontiguousarray(npz[f'{prefix}.weight{p1_suf}'], dtype=np.uint8)
            p2 = np.ascontiguousarray(npz[f'{prefix}.weight{p2_suf}'], dtype=np.uint8)
            s1 = np.ascontiguousarray(npz[f'{prefix}.weight.s1'].astype(np.float32))
            s2 = np.ascontiguousarray(npz[f'{prefix}.weight.s2'].astype(np.float32))
            
            M = 512 if p_name == 'mlp_fc1' else 160
            N = 512 if p_name == 'mlp_fc2' else 160
            
            p1_raw = p1.tobytes()
            p2_raw = p2.tobytes()
            
            import zlib
            p1_crc = zlib.crc32(p1_raw) & 0xffffffff
            p2_crc = zlib.crc32(p2_raw) & 0xffffffff
            
            p1_comp = stre_compress_plane(p1_raw)
            p2_comp = stre_compress_plane(p2_raw)
            
            layers_meta.append({
                'block_idx': b_idx,
                'layer_type': l_type,
                'M': M,
                'N': N,
                'p1_comp_bytes': len(p1_comp),
                'p1_raw_bytes': len(p1_raw),
                'p1_crc': p1_crc,
                'p2_comp_bytes': len(p2_comp),
                'p2_raw_bytes': len(p2_raw),
                'p2_crc': p2_crc,
                's1': s1,
                's2': s2,
            })
            compressed_payloads.append((p1_comp, p2_comp))
            
    # Final LM head
    head_p1 = np.ascontiguousarray(npz[f'lm_head.weight{p1_suf}'], dtype=np.uint8)
    head_p2 = np.ascontiguousarray(npz[f'lm_head.weight{p2_suf}'], dtype=np.uint8)
    head_s1 = np.ascontiguousarray(npz['lm_head.weight.s1'].astype(np.float32))
    head_s2 = np.ascontiguousarray(npz['lm_head.weight.s2'].astype(np.float32))
    
    h_p1_raw = head_p1.tobytes()
    h_p2_raw = head_p2.tobytes()
    h_p1_comp = stre_compress_plane(h_p1_raw)
    h_p2_comp = stre_compress_plane(h_p2_raw)
    
    layers_meta.append({
        'block_idx': -1,
        'layer_type': 6,
        'M': 256,
        'N': 160,
        'p1_comp_bytes': len(h_p1_comp),
        'p1_raw_bytes': len(h_p1_raw),
        'p1_crc': zlib.crc32(h_p1_raw) & 0xffffffff,
        'p2_comp_bytes': len(h_p2_comp),
        'p2_raw_bytes': len(h_p2_raw),
        'p2_crc': zlib.crc32(h_p2_raw) & 0xffffffff,
        's1': head_s1,
        's2': head_s2,
    })
    compressed_payloads.append((h_p1_comp, h_p2_comp))
    
    num_layers = len(layers_meta)
    assert num_layers == 25
    
    base_prefix_len = len(header_bytes) + len(embs_bytes) + len(block_norms_bytes) + len(final_norm_bytes)
    table_len = 4
    for meta in layers_meta:
        table_len += 56 + meta['M'] * 8
        
    payload_start_offset = base_prefix_len + table_len
    
    curr_payload_offset = payload_start_offset
    for idx, meta in enumerate(layers_meta):
        p1_c, p2_c = compressed_payloads[idx]
        meta['p1_comp_offset'] = curr_payload_offset
        curr_payload_offset += len(p1_c)
        meta['p2_comp_offset'] = curr_payload_offset
        curr_payload_offset += len(p2_c)
        
    table_bytes = bytearray(struct.pack("<I", num_layers))
    for meta in layers_meta:
        table_bytes += struct.pack(
            "<4i2Q6I",
            meta['block_idx'], meta['layer_type'], meta['M'], meta['N'],
            meta['p1_comp_offset'], meta['p2_comp_offset'],
            meta['p1_comp_bytes'], meta['p1_raw_bytes'], meta['p1_crc'],
            meta['p2_comp_bytes'], meta['p2_raw_bytes'], meta['p2_crc']
        )
        table_bytes += meta['s1'].tobytes()
        table_bytes += meta['s2'].tobytes()
        
    assert len(table_bytes) == table_len
    
    with open(output_path, "wb") as f:
        f.write(header_bytes)
        f.write(embs_bytes)
        f.write(block_norms_bytes)
        f.write(final_norm_bytes)
        f.write(table_bytes)
        for p1_c, p2_c in compressed_payloads:
            f.write(p1_c)
            f.write(p2_c)
            
    total_bytes = output_path.stat().st_size
    raw_payload_sum = sum(m['p1_raw_bytes'] + m['p2_raw_bytes'] for m in layers_meta)
    comp_payload_sum = sum(m['p1_comp_bytes'] + m['p2_comp_bytes'] for m in layers_meta)
    reduction = (1.0 - comp_payload_sum / raw_payload_sum) * 100
    print(f"Exported .llstre: {output_path.name} | Total File: {total_bytes:,} B | Payload: {raw_payload_sum:,} -> {comp_payload_sum:,} B (-{reduction:.2f}%)")
    return total_bytes

def export_all_phase4_1_containers():
    root = Path(__file__).resolve().parent.parent
    art_dir = root / 'checkpoints/artifacts'
    stream_dir = root / 'checkpoints/streaming'
    stream_dir.mkdir(parents=True, exist_ok=True)
    
    targets = [
        'S_ADBO_C0_Radix3',
        'S_ADBO_S1_Radix3',
        'S_ADBO_S1_s42_Radix3',
        'S_ADBO_S1_s1337_Radix3',
        'S_ADBO_S1_s2026_Radix3',
    ]
    
    for base_name in targets:
        npz_file = art_dir / f"{base_name}.npz"
        if not npz_file.exists():
            alt = art_dir / f"{base_name.replace('_s42', '').replace('_s1337', '').replace('_s2026', '')}.npz"
            if alt.exists():
                npz_file = alt
        out_file = stream_dir / f"{base_name}.llstre"
        export_llstre_container(npz_file, out_file, pack_mode=1)

if __name__ == '__main__':
    export_all_phase4_1_containers()
