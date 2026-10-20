"""
Generates publication-quality charts and SVG assets for LittleLess AI GitHub Research Portal.
Data drawn directly from empirical audited benchmarks in evidence/
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / 'docs/assets'
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# ---------------------------------------------------------------------------
# 1. Pareto Frontier: Storage vs Throughput
# ---------------------------------------------------------------------------
def plot_pareto():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    candidates = [
        ('Raw Streaming (Phase 3.9)', 572140, 4070.1, '#3b82f6', 'o', 100),
        ('zlib Chunk (Phase 4.0)', 443924, 987.1, '#ef4444', 's', 100),
        ('zlib + inflateReset (Exp A)', 443924, 1026.3, '#f97316', '^', 100),
        ('STRE + Zero-Skip (Phase 4.1 Champion)', 523756, 2029.6, '#10b981', '*', 200),
        ('SSTCD Rank-8 (Exp A)', 671016, 850.0, '#8b5cf6', 'X', 100),
    ]
    
    for label, size_b, tput, color, marker, s in candidates:
        ax.scatter(size_b / 1024, tput, color=color, marker=marker, s=s, label=label, zorder=5)
        offset_y = 60 if 'Champion' not in label else -120
        offset_x = 5
        ax.annotate(f"{label}\n({size_b/1024:.1f} KB, {tput:.0f} tok/s)",
                    (size_b / 1024, tput),
                    textcoords="offset points", xytext=(offset_x, offset_y),
                    fontsize=8, fontweight='bold' if 'Champion' in label else 'normal',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))

    # Acceptance Gate Lines
    ax.axhline(2000, color='#10b981', linestyle='--', alpha=0.7, label='Target Throughput Gate (>= 2,000 tok/s)')
    ax.axvline(380, color='#6366f1', linestyle=':', alpha=0.7, label='Target Storage Gate (< 380 KB)')
    
    ax.set_title('LittleLess AI — Storage vs Inference Throughput (Apple Silicon M4)', fontsize=12, fontweight='bold')
    ax.set_xlabel('On-Disk Artifact Size (KB)', fontsize=11)
    ax.set_ylabel('Decode Throughput (tokens/sec)', fontsize=11)
    ax.set_xlim(350, 750)
    ax.set_ylim(500, 4500)
    ax.legend(loc='upper right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'storage_vs_throughput.png')
    plt.savefig(ASSETS_DIR / 'storage_vs_throughput.svg')
    plt.close()
    print("Saved storage_vs_throughput.[png|svg]")

# ---------------------------------------------------------------------------
# 2. Native Model Heap RAM Breakdown
# ---------------------------------------------------------------------------
def plot_memory_breakdown():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    modes = ['Static Model\n(Phase 3.5)', 'Raw Stream\nFP32 KV (P3.9)', 'Raw Stream\nInt8 KV (P4.0)', 'STRE Coalesced\nInt8 KV (P4.1)']
    active_weights = np.array([570140, 160704, 160704, 193664]) / 1024 # KB
    kv_cache = np.array([655360, 655360, 184320, 184320]) / 1024 # KB
    activations = np.array([8320, 8320, 8320, 8320]) / 1024 # KB
    
    x = np.arange(len(modes))
    width = 0.55
    
    b1 = ax.bar(x, active_weights, width, label='Active Weight RAM (Scratchpad + Embeddings)', color='#3b82f6')
    b2 = ax.bar(x, kv_cache, width, bottom=active_weights, label='KV Cache RAM (128 context)', color='#10b981')
    b3 = ax.bar(x, activations, width, bottom=active_weights + kv_cache, label='Activations & Residuals', color='#f59e0b')
    
    totals = active_weights + kv_cache + activations
    for i, tot in enumerate(totals):
        ax.text(i, tot + 15, f"{tot:.1f} KB\n(-{(1 - tot/totals[0])*100:.1f}%)" if i > 0 else f"{tot:.1f} KB",
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    ax.set_title('LittleLess AI — Native C++ Heap RAM Footprint Reduction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Native Heap RAM (KB)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=9)
    ax.set_ylim(0, 1500)
    ax.legend(loc='upper right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'memory_breakdown.png')
    plt.savefig(ASSETS_DIR / 'memory_breakdown.svg')
    plt.close()
    print("Saved memory_breakdown.[png|svg]")

# ---------------------------------------------------------------------------
# 3. Profiling Latency Breakdown
# ---------------------------------------------------------------------------
def plot_latency_breakdown():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    categories = ['Decompression', 'Pread Syscalls', 'GEMV Execution', 'Int8 KV Update', 'LayerNorm/Activations']
    raw_times = [0.0, 148.2, 128.5, 34.8, 49.2] # us
    zlib_times = [782.4, 114.6, 128.5, 34.8, 49.2] # us
    stre_times = [75.8, 72.3, 98.2, 34.8, 49.2] # us
    
    x = np.arange(len(categories))
    w = 0.25
    
    ax.bar(x - w, raw_times, w, label='Raw File-Streaming (360.7 us)', color='#3b82f6')
    ax.bar(x, zlib_times, w, label='Phase 4.0 zlib Chunk (1,109.5 us)', color='#ef4444')
    ax.bar(x + w, stre_times, w, label='Phase 4.1 STRE + Fused (330.3 us)', color='#10b981')
    
    ax.set_title('Native Execution Breakdown per Forward Step (Apple Silicon M4)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (microseconds / token)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)
    ax.legend(loc='upper right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'profiling_breakdown.png')
    plt.savefig(ASSETS_DIR / 'profiling_breakdown.svg')
    plt.close()
    print("Saved profiling_breakdown.[png|svg]")

if __name__ == '__main__':
    plot_pareto()
    plot_memory_breakdown()
    plot_latency_breakdown()
