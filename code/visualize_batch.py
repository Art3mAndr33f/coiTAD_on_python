#!/usr/bin/env python3
"""Post-hoc visualization for multi-dataset batch results."""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from pathlib import Path
from itertools import product

from config import (DATASETS, CHROMOSOMES, RESOLUTIONS, METHODS,
                    COLORS, BATCH_OUTPUT as ROOT)


def find_tad_file(exp_dir, method):
    tad_dir = exp_dir / f"results_{method}" / "TADs"
    if not tad_dir.exists(): return None
    qf = exp_dir / f"results_{method}" / "Quality" / "Readme.txt"
    if qf.exists():
        for part in qf.read_text().split():
            if f"{method}_" in part and "_domain" in part:
                r = part.replace(f"{method}_", "").replace("_domain.txt", "")
                f = tad_dir / f"{method}_{r}_TAD_BinID.txt"
                if f.exists(): return f
    candidates = sorted(tad_dir.glob(f"{method}_*_TAD_BinID.txt"))
    return candidates[-1] if candidates else None


def load_tads(f):
    try:
        d = np.loadtxt(str(f), dtype=int)
        return d.reshape(1, -1) if d.ndim == 1 else d
    except: return np.empty((0, 2), dtype=int)


def find_unique(main, other, tol=2):
    if len(main) == 0: return np.empty((0, 2), dtype=int)
    if len(other) == 0: return main.copy()
    u = [t for t in main if not any(
        abs(t[0]-o[0]) <= tol and abs(t[1]-o[1]) <= tol for o in other)]
    return np.array(u, dtype=int) if u else np.empty((0, 2), dtype=int)


def _set_axes(ax, n, res):
    step = max(1, n // 8); t = np.arange(0, n, step)
    ax.set_xticks(t); ax.set_yticks(t)
    ax.set_xticklabels([f'{x*res/1e6:.1f}' for x in t], fontsize=9)
    ax.set_yticklabels([f'{x*res/1e6:.1f}' for x in t], fontsize=9)
    ax.set_xlabel('Position (Mb)'); ax.set_ylabel('Position (Mb)')


def plot_overlay(matrix, tads_dict, ds, chrom, res, out_dir):
    fig, ax = plt.subplots(figsize=(11, 10))
    m = matrix.copy(); m[m == 0] = np.nan
    ax.imshow(m, cmap='Reds', norm=LogNorm(), interpolation='none', origin='upper')
    handles = []
    for method, tads in tads_dict.items():
        c = COLORS.get(method, 'green')
        ls = '-' if method == 'HDBSCAN' else '--'
        for s, e in tads:
            ax.add_patch(patches.Rectangle((s-.5, s-.5), e-s+1, e-s+1,
                         lw=1.6, edgecolor=c, facecolor='none', alpha=0.7, linestyle=ls))
        handles.append(patches.Patch(edgecolor=c, facecolor='none', linestyle=ls,
                                     lw=2, label=f'{method} ({len(tads)})'))
    ax.legend(handles=handles, loc='upper right', fontsize=11)
    _set_axes(ax, matrix.shape[0], res)
    ax.set_title(f'{ds} / {chrom} @ {res//1000}kb', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"{chrom}_{res//1000}kb_overlay.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: overlay")


def plot_unique_tads(matrix, tads_h, tads_o, ds, chrom, res, out_dir, tol=2):
    optics_only = find_unique(tads_o, tads_h, tol)
    hdbscan_only = find_unique(tads_h, tads_o, tol)
    n_shared = len(tads_o) - len(optics_only)
    print(f"  Shared: {n_shared}, OPTICS-only: {len(optics_only)}, HDBSCAN-only: {len(hdbscan_only)}")
    if len(optics_only) == 0: return

    fig, ax = plt.subplots(figsize=(12, 10))
    m = matrix.copy(); m[m == 0] = np.nan
    ax.imshow(m, cmap='Reds', norm=LogNorm(), interpolation='none', origin='upper')
    for s, e in tads_o:
        if not any(s == u[0] and e == u[1] for u in optics_only):
            ax.add_patch(patches.Rectangle((s-.5, s-.5), e-s+1, e-s+1,
                         lw=1, edgecolor='grey', facecolor='none', alpha=0.4, linestyle=':'))
    for s, e in optics_only:
        ax.add_patch(patches.Rectangle((s-.5, s-.5), e-s+1, e-s+1,
                     lw=2.5, edgecolor='red', facecolor='red', alpha=0.08))
        ax.add_patch(patches.Rectangle((s-.5, s-.5), e-s+1, e-s+1,
                     lw=2.5, edgecolor='red', facecolor='none', alpha=0.9))
        c = (s+e)/2; kb = (e-s)*res/1000
        ax.text(c, c, f'{kb:.0f}kb', ha='center', va='center', fontsize=8,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.85))
    for s, e in hdbscan_only:
        ax.add_patch(patches.Rectangle((s-.5, s-.5), e-s+1, e-s+1,
                     lw=1.5, edgecolor='blue', facecolor='none', alpha=0.5, linestyle='--'))
    items = [
        patches.Patch(facecolor='red', alpha=0.3, edgecolor='red', lw=2, label=f'OPTICS-only ({len(optics_only)})'),
        patches.Patch(facecolor='none', edgecolor='grey', linestyle=':', lw=1, label=f'Shared ({n_shared})'),
        patches.Patch(facecolor='none', edgecolor='blue', linestyle='--', lw=1.5, label=f'HDBSCAN-only ({len(hdbscan_only)})'),
    ]
    ax.legend(handles=items, loc='upper right', fontsize=11)
    _set_axes(ax, matrix.shape[0], res)
    ax.set_title(f'{ds} / {chrom} @ {res//1000}kb - OPTICS-only (tol={tol})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"{chrom}_{res//1000}kb_optics_only.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: optics_only")

    # Zoom panels
    nc = min(4, len(optics_only)); nr = int(np.ceil(len(optics_only)/nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if nr == 1 and nc == 1: axes = np.array([[axes]])
    elif nr == 1: axes = axes.reshape(1, -1)
    elif nc == 1: axes = axes.reshape(-1, 1)
    ctx = 10; nb = matrix.shape[0]
    for idx, (s, e) in enumerate(optics_only):
        r, c = divmod(idx, nc); ax = axes[r, c]
        s0, e0 = max(0, s-ctx), min(nb, e+ctx+1)
        sub = matrix[s0:e0, s0:e0].copy(); sub[sub == 0] = np.nan
        ax.imshow(sub, cmap='YlOrRd', interpolation='none', origin='upper')
        ax.add_patch(patches.Rectangle((s-s0-.5, s-s0-.5), e-s+1, e-s+1,
                     lw=2.5, edgecolor='red', facecolor='none'))
        for hs, he in tads_h:
            if he >= s0 and hs <= e0:
                rhs, rhe = max(hs-s0, 0), min(he-s0, e0-s0-1)
                ax.add_patch(patches.Rectangle((rhs-.5, rhs-.5), rhe-rhs+1, rhe-rhs+1,
                             lw=1.5, edgecolor='blue', facecolor='none', linestyle='--', alpha=0.6))
        ax.set_title(f'#{idx+1} {(e-s)*res/1000:.0f}kb', fontsize=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
    for idx in range(len(optics_only), nr*nc):
        r, c = divmod(idx, nc); axes[r, c].axis('off')
    fig.suptitle(f'{ds} / {chrom} @ {res//1000}kb - OPTICS-only zoomed', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / f"{chrom}_{res//1000}kb_optics_only_zoom.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: optics_only_zoom")


def plot_summary_bar(df, out_dir):
    if df is None or len(df) == 0: return
    fig, ax = plt.subplots(figsize=(max(14, len(df)*0.4), 7))
    df = df.copy()
    df['label'] = df['dataset']+'\n'+df['chromosome']+'\n'+df['resolution_kb'].astype(str)+'kb'
    labels = df.drop_duplicates(subset=['dataset','chromosome','resolution_kb'])['label'].values
    x = np.arange(len(labels)); w = 0.35
    for i, method in enumerate(METHODS):
        sub = df[df['method'] == method]
        counts = [int(sub[sub['label']==l]['n_tads'].values[0])
                  if len(sub[sub['label']==l]) > 0 else 0 for l in labels]
        bars = ax.bar(x+(i-.5)*w, counts, w, label=method,
                      color=COLORS.get(method,'gray'), edgecolor='black')
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.5,
                        str(int(b.get_height())), ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('TADs'); ax.set_title('TAD Counts Across Datasets', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir/"summary_tad_counts.png", dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved: summary_tad_counts")


def main():
    viz_root = ROOT / "visualizations"; viz_root.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT/"all_results.csv") if (ROOT/"all_results.csv").exists() else None

    for ds_name in DATASETS:
        for chrom, res in product(CHROMOSOMES, RESOLUTIONS):
            exp_dir = ROOT / ds_name / f"{chrom}_{res//1000}kb"
            if not exp_dir.exists(): continue
            mf = exp_dir / "data" / f"{chrom}_{res//1000}kb.hic"
            if not mf.exists(): continue
            matrix = np.loadtxt(str(mf))
            print(f"\n--- {ds_name}/{chrom} @ {res//1000}kb ---")
            tads_dict = {}
            for method in METHODS:
                tf = find_tad_file(exp_dir, method)
                if tf:
                    t = load_tads(tf)
                    if len(t) > 0: tads_dict[method] = t
            if not tads_dict: continue
            cond = viz_root / ds_name / f"{chrom}_{res//1000}kb"
            cond.mkdir(parents=True, exist_ok=True)
            if len(tads_dict) >= 2:
                plot_overlay(matrix, tads_dict, ds_name, chrom, res, cond)
            if "HDBSCAN" in tads_dict and "OPTICS" in tads_dict:
                plot_unique_tads(matrix, tads_dict["HDBSCAN"], tads_dict["OPTICS"],
                                 ds_name, chrom, res, cond)
    if df is not None: plot_summary_bar(df, viz_root)
    print(f"\nDone: {viz_root}/")


if __name__ == "__main__":
    main()