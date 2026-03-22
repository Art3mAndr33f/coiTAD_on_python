#!/usr/bin/env python3
"""
Post-hoc visualization for existing batch results.
Does NOT re-run any TAD calling — reads saved files only.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from pathlib import Path
from itertools import product
import pandas as pd


# ======================================================================
# Config — должен совпадать с run_batch.py
# ======================================================================

ROOT_OUTPUT = Path("batch_results")

CHROMOSOMES = ["chr19", "chr17", "chr22"]
RESOLUTIONS = [25000, 50000, 100000]
METHODS = ["HDBSCAN", "OPTICS"]

COLORS = {"HDBSCAN": "blue", "OPTICS": "red"}
METHOD_LS = {"HDBSCAN": "-", "OPTICS": "--"}


# ======================================================================
# Helpers
# ======================================================================

def find_tad_file(results_dir: Path, method: str) -> Path | None:
    """Find the best-radius TAD_BinID file for a method."""
    tad_dir = results_dir / f"results_{method}" / "TADs"
    if not tad_dir.exists():
        return None

    # Look for *_TAD_BinID.txt files
    candidates = sorted(tad_dir.glob(f"{method}_*_TAD_BinID.txt"))
    if not candidates:
        return None

    # Pick the one from Readme (best radius)
    quality_dir = results_dir / f"results_{method}" / "Quality"
    if quality_dir.exists():
        readme = quality_dir / "Readme.txt"
        if readme.exists():
            text = readme.read_text()
            # parse "Recommended TAD = HDBSCAN_5_domain.txt ..."
            for part in text.split():
                if f"{method}_" in part and "_domain" in part:
                    radius = part.replace(f"{method}_", "").replace("_domain.txt", "")
                    best_file = tad_dir / f"{method}_{radius}_TAD_BinID.txt"
                    if best_file.exists():
                        return best_file

    # Fallback: last file (highest radius that was saved)
    return candidates[-1]


def load_tads(tad_file: Path) -> np.ndarray:
    """Load TAD boundaries, return (N, 2) array."""
    try:
        data = np.loadtxt(str(tad_file), dtype=int)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception:
        return np.empty((0, 2), dtype=int)


def load_matrix(data_dir: Path, chrom: str, res: int) -> np.ndarray | None:
    """Load contact matrix."""
    matrix_file = data_dir / f"{chrom}_{res // 1000}kb.hic"
    if not matrix_file.exists():
        return None
    return np.loadtxt(str(matrix_file))


# ======================================================================
# 1. Contact map with TADs from BOTH methods side by side
# ======================================================================

def plot_side_by_side(matrix, tads_dict, chrom, res, output_dir):
    """Two panels: HDBSCAN TADs | OPTICS TADs on same contact map."""
    n_methods = len(tads_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(10 * n_methods, 9))
    if n_methods == 1:
        axes = [axes]

    mat_plot = matrix.copy()
    mat_plot[mat_plot == 0] = np.nan

    for ax, (method, tads) in zip(axes, tads_dict.items()):
        im = ax.imshow(mat_plot, cmap='Reds', norm=LogNorm(),
                       interpolation='none', origin='upper')

        color = COLORS.get(method, 'blue')
        for s, e in tads:
            rect = patches.Rectangle(
                (s - 0.5, s - 0.5), e - s + 1, e - s + 1,
                linewidth=1.8, edgecolor=color,
                facecolor='none', alpha=0.8)
            ax.add_patch(rect)

        ax.set_title(f'{method}  ({len(tads)} TADs)',
                     fontsize=13, fontweight='bold')

        # Axis labels in Mb
        n = matrix.shape[0]
        step = max(1, n // 8)
        ticks = np.arange(0, n, step)
        labels = [f'{t * res / 1e6:.1f}' for t in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Position (Mb)')
        ax.set_ylabel('Position (Mb)')

    fig.suptitle(f'{chrom}  @ {res // 1000} kb',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label='Contact Frequency')

    plt.tight_layout()
    out = output_dir / f"{chrom}_{res // 1000}kb_side_by_side.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
# 2. Overlay — both methods on one map (upper triangle / lower triangle)
# ======================================================================

def plot_overlay(matrix, tads_dict, chrom, res, output_dir):
    """Single map, HDBSCAN rectangles + OPTICS rectangles in diff colors."""
    fig, ax = plt.subplots(figsize=(11, 10))

    mat_plot = matrix.copy()
    mat_plot[mat_plot == 0] = np.nan

    ax.imshow(mat_plot, cmap='Reds', norm=LogNorm(),
              interpolation='none', origin='upper')

    legend_handles = []
    for method, tads in tads_dict.items():
        color = COLORS.get(method, 'green')
        ls = METHOD_LS.get(method, '-')
        for s, e in tads:
            rect = patches.Rectangle(
                (s - 0.5, s - 0.5), e - s + 1, e - s + 1,
                linewidth=1.6, edgecolor=color,
                facecolor='none', alpha=0.7, linestyle=ls)
            ax.add_patch(rect)
        # Legend proxy
        legend_handles.append(
            patches.Patch(edgecolor=color, facecolor='none',
                          linestyle=ls, linewidth=2,
                          label=f'{method} ({len(tads)} TADs)'))

    ax.legend(handles=legend_handles, loc='upper right', fontsize=11,
              framealpha=0.9)

    n = matrix.shape[0]
    step = max(1, n // 8)
    ticks = np.arange(0, n, step)
    labels = [f'{t * res / 1e6:.1f}' for t in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Position (Mb)')
    ax.set_ylabel('Position (Mb)')
    ax.set_title(f'{chrom} @ {res // 1000} kb — HDBSCAN vs OPTICS overlay',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    out = output_dir / f"{chrom}_{res // 1000}kb_overlay.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
# 3. TAD size distribution comparison
# ======================================================================

def plot_size_distributions(all_tads, res, output_dir):
    """
    all_tads: dict  (chrom, method) -> tads array
    One figure per resolution with subplots per chromosome.
    """
    chroms = sorted({k[0] for k in all_tads})
    n_chr = len(chroms)
    if n_chr == 0:
        return

    fig, axes = plt.subplots(1, n_chr, figsize=(6 * n_chr, 5))
    if n_chr == 1:
        axes = [axes]

    for ax, chrom in zip(axes, chroms):
        for method in METHODS:
            key = (chrom, method)
            if key not in all_tads or len(all_tads[key]) == 0:
                continue
            tads = all_tads[key]
            sizes = (tads[:, 1] - tads[:, 0]) * res / 1000
            ax.hist(sizes, bins=20, alpha=0.5, label=method,
                    edgecolor='black', color=COLORS.get(method, 'gray'))

        ax.set_xlabel('TAD Size (kb)')
        ax.set_ylabel('Count')
        ax.set_title(f'{chrom}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'TAD Size Distributions @ {res // 1000} kb',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = output_dir / f"size_distributions_{res // 1000}kb.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
# 4. Summary bar chart (n_tads across all conditions)
# ======================================================================

def plot_summary_bar(df_results, output_dir):
    """Grouped bar chart: TAD counts per condition."""
    if df_results is None or len(df_results) == 0:
        print("  No results CSV found, skipping summary bar.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(df_results) * 0.8), 6))

    df_results['label'] = (df_results['chromosome'] + '\n' +
                           df_results['resolution_kb'].astype(str) + 'kb')

    labels_unique = df_results.drop_duplicates(
        subset=['chromosome', 'resolution_kb'])['label'].values
    x = np.arange(len(labels_unique))
    width = 0.35

    for i, method in enumerate(METHODS):
        sub = df_results[df_results['method'] == method]
        # Align by label order
        counts = []
        for lbl in labels_unique:
            row = sub[sub['label'] == lbl]
            counts.append(int(row['n_tads'].values[0]) if len(row) > 0 else 0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, counts, width, label=method,
                      color=COLORS.get(method, 'gray'), edgecolor='black')
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                        str(int(h)), ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_unique, fontsize=10)
    ax.set_ylabel('Number of TADs')
    ax.set_title('TAD Counts: HDBSCAN vs OPTICS', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = output_dir / "summary_tad_counts.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
# 5. Enrichment comparison bar chart
# ======================================================================

def plot_enrichment_summary(df_results, output_dir):
    """Bar chart: avg enrichment per marker, grouped by method."""
    marker_cols = [c for c in df_results.columns
                   if c.startswith("avg_") and c.endswith("_peaks")]
    if not marker_cols:
        print("  No enrichment columns found, skipping.")
        return

    markers = [c.replace("avg_", "").replace("_peaks", "") for c in marker_cols]

    fig, ax = plt.subplots(figsize=(max(8, len(markers) * 3), 5))

    x = np.arange(len(markers))
    width = 0.35

    for i, method in enumerate(METHODS):
        sub = df_results[df_results['method'] == method]
        means = [sub[mc].mean() for mc in marker_cols]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, label=method,
                      color=COLORS.get(method, 'gray'), edgecolor='black')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(markers, fontsize=11)
    ax.set_ylabel('Avg peaks / bin at TAD boundaries')
    ax.set_title('Biological Validation Summary (all chromosomes & resolutions)',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = output_dir / "summary_enrichment.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
# 6. Concordance metrics heatmap across conditions
# ======================================================================

def plot_concordance_heatmap(df_comp, output_dir):
    """Heatmap: MoC / ARI / NMI across conditions."""
    if df_comp is None or len(df_comp) == 0:
        print("  No comparisons CSV found, skipping concordance heatmap.")
        return

    df_comp['label'] = (df_comp['chromosome'] + ' ' +
                        df_comp['resolution_kb'].astype(str) + 'kb')

    metrics = ['MoC', 'ARI', 'NMI']
    available = [m for m in metrics if m in df_comp.columns]

    fig, axes = plt.subplots(1, len(available),
                             figsize=(5 * len(available), max(4, len(df_comp) * 0.6)))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        data = df_comp.set_index('label')[[metric]]
        im = ax.imshow(data.values, cmap='YlGn', vmin=0, vmax=1, aspect='auto')

        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, fontsize=10)
        ax.set_xticks([0])
        ax.set_xticklabels([metric], fontsize=12)

        for i, val in enumerate(data.values.flatten()):
            ax.text(0, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold')

        ax.set_title(metric, fontsize=13, fontweight='bold')

    fig.suptitle('HDBSCAN vs OPTICS Concordance',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = output_dir / "summary_concordance.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

# ======================================================================
# 7. OPTICS-only TADs on contact map
# ======================================================================

def plot_unique_tads(matrix, tads_hdb, tads_opt,
                     chrom, res, output_dir, tolerance=2):
    """
    Highlight TADs found by OPTICS but missed by HDBSCAN.

    - Grey rectangles  = shared TADs (matched in both methods)
    - Red rectangles   = OPTICS-only TADs
    - Annotations with TAD size in kb
    """
    optics_only = find_unique_tads(tads_opt, tads_hdb, tolerance)
    hdbscan_only = find_unique_tads(tads_hdb, tads_opt, tolerance)

    # Shared = OPTICS TADs that DO have a match in HDBSCAN
    shared = find_unique_tads(tads_opt, optics_only, tolerance=0)

    n_shared = len(tads_opt) - len(optics_only)

    print(f"  OPTICS total: {len(tads_opt)}")
    print(f"  HDBSCAN total: {len(tads_hdb)}")
    print(f"  Shared (tol={tolerance}): {n_shared}")
    print(f"  OPTICS-only: {len(optics_only)}")
    print(f"  HDBSCAN-only: {len(hdbscan_only)}")

    if len(optics_only) == 0:
        print("  No OPTICS-only TADs found — skipping plot.")
        return

    # ---- Full map ----

    fig, ax = plt.subplots(figsize=(12, 10))

    mat_plot = matrix.copy()
    mat_plot[mat_plot == 0] = np.nan
    ax.imshow(mat_plot, cmap='Reds', norm=LogNorm(),
              interpolation='none', origin='upper')

    # Draw shared TADs (grey, thin)
    for s, e in tads_opt:
        is_unique = any((s == u[0] and e == u[1]) for u in optics_only)
        if not is_unique:
            rect = patches.Rectangle(
                (s - 0.5, s - 0.5), e - s + 1, e - s + 1,
                linewidth=1.0, edgecolor='grey',
                facecolor='none', alpha=0.4, linestyle=':')
            ax.add_patch(rect)

    # Draw OPTICS-only TADs (red, bold + annotated)
    for idx, (s, e) in enumerate(optics_only):
        size_kb = (e - s) * res / 1000

        rect = patches.Rectangle(
            (s - 0.5, s - 0.5), e - s + 1, e - s + 1,
            linewidth=2.5, edgecolor='red',
            facecolor='red', alpha=0.08)
        ax.add_patch(rect)

        # Bold border
        rect2 = patches.Rectangle(
            (s - 0.5, s - 0.5), e - s + 1, e - s + 1,
            linewidth=2.5, edgecolor='red',
            facecolor='none', alpha=0.9)
        ax.add_patch(rect2)

        # Label
        center = (s + e) / 2
        ax.text(center, center, f'{size_kb:.0f}kb',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='red', alpha=0.85, edgecolor='none'))

    # Draw HDBSCAN-only TADs (blue dashed, for reference)
    for s, e in hdbscan_only:
        rect = patches.Rectangle(
            (s - 0.5, s - 0.5), e - s + 1, e - s + 1,
            linewidth=1.5, edgecolor='blue',
            facecolor='none', alpha=0.5, linestyle='--')
        ax.add_patch(rect)

    # Legend
    legend_items = [
        patches.Patch(facecolor='red', alpha=0.3, edgecolor='red',
                      linewidth=2, label=f'OPTICS-only ({len(optics_only)})'),
        patches.Patch(facecolor='none', edgecolor='grey',
                      linestyle=':', linewidth=1,
                      label=f'Shared ({n_shared})'),
        patches.Patch(facecolor='none', edgecolor='blue',
                      linestyle='--', linewidth=1.5,
                      label=f'HDBSCAN-only ({len(hdbscan_only)})'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=11,
              framealpha=0.9)

    # Axes
    n = matrix.shape[0]
    step = max(1, n // 8)
    ticks = np.arange(0, n, step)
    labels = [f'{t * res / 1e6:.1f}' for t in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Position (Mb)')
    ax.set_ylabel('Position (Mb)')
    ax.set_title(
        f'{chrom} @ {res // 1000} kb — TADs unique to OPTICS\n'
        f'(tolerance = {tolerance} bins = {tolerance * res / 1000:.0f} kb)',
        fontsize=13, fontweight='bold')

    plt.tight_layout()
    out = output_dir / f"{chrom}_{res // 1000}kb_optics_only.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # ---- Zoomed panels for each OPTICS-only TAD ----

    if len(optics_only) > 0:
        plot_unique_tads_zoom(matrix, optics_only, tads_hdb,
                              chrom, res, output_dir, tolerance)


# ======================================================================
# 7b. Zoomed view of each OPTICS-only TAD
# ======================================================================

def plot_unique_tads_zoom(matrix, unique_tads, other_tads,
                          chrom, res, output_dir, tolerance=2):
    """Grid of zoomed sub-matrices for each OPTICS-only TAD."""
    n_tads = len(unique_tads)
    if n_tads == 0:
        return

    n_cols = min(4, n_tads)
    n_rows = int(np.ceil(n_tads / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    context = 10  # bins of context around TAD
    n_bins = matrix.shape[0]

    for idx, (s, e) in enumerate(unique_tads):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        s0 = max(0, s - context)
        e0 = min(n_bins, e + context + 1)
        sub = matrix[s0:e0, s0:e0].copy()
        sub[sub == 0] = np.nan

        ax.imshow(sub, cmap='YlOrRd', interpolation='none', origin='upper')

        # OPTICS-only TAD (red)
        rs, re = s - s0, e - s0
        rect = patches.Rectangle(
            (rs - 0.5, rs - 0.5), re - rs + 1, re - rs + 1,
            linewidth=2.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Nearest HDBSCAN TADs in this region (blue dashed)
        for hs, he in other_tads:
            if he >= s0 and hs <= e0:
                rhs = max(hs - s0, 0)
                rhe = min(he - s0, e0 - s0 - 1)
                rect_h = patches.Rectangle(
                    (rhs - 0.5, rhs - 0.5), rhe - rhs + 1, rhe - rhs + 1,
                    linewidth=1.5, edgecolor='blue', facecolor='none',
                    linestyle='--', alpha=0.6)
                ax.add_patch(rect_h)

        size_kb = (e - s) * res / 1000
        pos_mb_start = s * res / 1e6
        pos_mb_end = e * res / 1e6
        ax.set_title(f'#{idx + 1}  {size_kb:.0f} kb\n'
                     f'{pos_mb_start:.2f}–{pos_mb_end:.2f} Mb',
                     fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_tads, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis('off')

    fig.suptitle(
        f'{chrom} @ {res // 1000} kb — OPTICS-only TADs (zoomed)\n'
        f'Red = OPTICS-only, Blue dashed = nearest HDBSCAN TADs',
        fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    out = output_dir / f"{chrom}_{res // 1000}kb_optics_only_zoom.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

# ======================================================================
# Helper: find TADs unique to one method
# ======================================================================

def find_unique_tads(tads_main: np.ndarray,
                     tads_other: np.ndarray,
                     tolerance: int = 2) -> np.ndarray:
    """
    Return TADs from tads_main that have NO match in tads_other.

    A match = both start and end within ±tolerance bins.

    Args:
        tads_main:  TADs to check  (N, 2)
        tads_other: TADs to compare against (M, 2)
        tolerance:  allowed boundary shift in bins

    Returns:
        Subset of tads_main that are unique (K, 2)
    """
    if len(tads_main) == 0:
        return np.empty((0, 2), dtype=int)
    if len(tads_other) == 0:
        return tads_main.copy()

    unique = []
    for s, e in tads_main:
        matched = False
        for s2, e2 in tads_other:
            if abs(s - s2) <= tolerance and abs(e - e2) <= tolerance:
                matched = True
                break
        if not matched:
            unique.append([s, e])

    return np.array(unique, dtype=int) if unique else np.empty((0, 2), dtype=int)

# ======================================================================
# Main
# ======================================================================

def main():
    viz_dir = ROOT_OUTPUT / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Load aggregated CSVs (if they exist)
    results_csv = ROOT_OUTPUT / "all_results.csv"
    comp_csv = ROOT_OUTPUT / "all_comparisons.csv"

    df_results = pd.read_csv(results_csv) if results_csv.exists() else None
    df_comp = pd.read_csv(comp_csv) if comp_csv.exists() else None

    print("=" * 70)
    print("POST-HOC VISUALIZATION (no re-running)")
    print("=" * 70)

    # ---- Per-condition plots (contact maps) ----
    size_data_by_res = {}  # res -> { (chrom, method): tads }

    for chrom, res in product(CHROMOSOMES, RESOLUTIONS):
        exp_dir = ROOT_OUTPUT / f"{chrom}_{res // 1000}kb"
        if not exp_dir.exists():
            print(f"\n  Skip {chrom} @ {res // 1000}kb — dir not found")
            continue

        data_dir = exp_dir / "data"
        matrix = load_matrix(data_dir, chrom, res)
        if matrix is None:
            print(f"\n  Skip {chrom} @ {res // 1000}kb — no matrix")
            continue

        print(f"\n--- {chrom} @ {res // 1000}kb ---")

        tads_dict = {}
        for method in METHODS:
            tad_file = find_tad_file(exp_dir, method)
            if tad_file is not None:
                tads = load_tads(tad_file)
                if len(tads) > 0:
                    tads_dict[method] = tads

                    # Accumulate for size distributions
                    if res not in size_data_by_res:
                        size_data_by_res[res] = {}
                    size_data_by_res[res][(chrom, method)] = tads

        if len(tads_dict) == 0:
            print("  No TAD files found")
            continue

        # Per-condition visualizations
        cond_viz = viz_dir / f"{chrom}_{res // 1000}kb"
        cond_viz.mkdir(exist_ok=True)

        plot_side_by_side(matrix, tads_dict, chrom, res, cond_viz)
        if len(tads_dict) >= 2:
            plot_overlay(matrix, tads_dict, chrom, res, cond_viz)

        if "HDBSCAN" in tads_dict and "OPTICS" in tads_dict:
            plot_unique_tads(matrix,
                             tads_dict["HDBSCAN"],
                             tads_dict["OPTICS"],
                             chrom, res, cond_viz,
                             tolerance=2)

    # ---- Aggregated plots ----
    print("\n--- Aggregated plots ---")

    for res, tads_data in size_data_by_res.items():
        plot_size_distributions(tads_data, res, viz_dir)

    if df_results is not None:
        plot_summary_bar(df_results, viz_dir)
        plot_enrichment_summary(df_results, viz_dir)

    if df_comp is not None:
        plot_concordance_heatmap(df_comp, viz_dir)

    print(f"\n{'=' * 70}")
    print(f"All visualizations saved to: {viz_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()