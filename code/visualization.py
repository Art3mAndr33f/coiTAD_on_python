"""
Visualization of coiTAD results.
Contact maps, TAD boundaries, statistics, interactive mcool view.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Tuple, Optional
import seaborn as sns


class CoiTADVisualizer:
    """Full-featured TAD / Hi-C visualizer"""

    def __init__(self,
                 contact_matrix_file: str,
                 tad_file: str,
                 resolution: int = 40000,
                 output_dir: str = "visualizations"):
        self.contact_matrix = np.loadtxt(contact_matrix_file)
        self.tad_borders = self._load_tad_borders(tad_file)
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tad_borders(tad_file: str) -> List[Tuple[int, int]]:
        try:
            data = np.loadtxt(tad_file, dtype=int)
        except ValueError:
            data = np.loadtxt(tad_file, skiprows=1, usecols=(0, 2), dtype=int)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return [(int(r[0]), int(r[1])) for r in data]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_tad_rectangles(self, ax, borders,
                             color='blue', linewidth=2, alpha=0.8):
        for start, end in borders:
            rect = patches.Rectangle(
                (start - 0.5, start - 0.5),
                end - start + 1, end - start + 1,
                linewidth=linewidth, edgecolor=color,
                facecolor='none', alpha=alpha)
            ax.add_patch(rect)

    def _setup_axes(self, ax, n_bins):
        step = max(1, n_bins // 10)
        ticks = np.arange(0, n_bins, step)
        labels = [f'{pos * self.resolution / 1_000_000:.1f}' for pos in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Genomic Position (Mb)', fontsize=12)
        ax.set_ylabel('Genomic Position (Mb)', fontsize=12)
        ax.grid(False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_contact_map_with_tads(self, figsize=(12, 10), cmap='Reds',
                                   log_scale=True, vmin=None, vmax=None,
                                   title=None,
                                   save_name="contact_map_with_tads.png",
                                   dpi=300):
        fig, ax = plt.subplots(figsize=figsize)
        matrix = self.contact_matrix.copy()
        matrix[matrix == 0] = np.nan

        norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
        kw = dict(cmap=cmap, interpolation='none', origin='upper')
        if log_scale:
            kw['norm'] = norm
        else:
            kw.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(matrix, **kw)

        self._draw_tad_rectangles(ax, self.tad_borders)
        self._setup_axes(ax, matrix.shape[0])

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Contact Frequency', rotation=270, labelpad=20)
        ax.set_title(
            title or f'Hi-C Contact Map with TAD Boundaries\n'
                      f'({len(self.tad_borders)} TADs detected)',
            fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        out = self.output_dir / save_name
        plt.savefig(out, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.show()

    def plot_tad_zoom(self, tad_index=0, context_bins=10,
                      figsize=(10, 8), cmap='RdYlBu_r', save_name=None):
        if tad_index >= len(self.tad_borders):
            print(f"TAD index {tad_index} out of range ({len(self.tad_borders)} TADs)")
            return

        start, end = self.tad_borders[tad_index]
        s = max(0, start - context_bins)
        e = min(self.contact_matrix.shape[0], end + context_bins)
        sub = self.contact_matrix[s:e, s:e]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(sub, cmap=cmap, interpolation='none', origin='upper')

        rs, re = start - s, end - s
        rect = patches.Rectangle(
            (rs - 0.5, rs - 0.5), re - rs + 1, re - rs + 1,
            linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)

        size_kb = (end - start) * self.resolution / 1000
        ax.set_title(f'TAD #{tad_index + 1}  |  bins {start}-{end}  |  '
                     f'{size_kb:.1f} kb', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        save_name = save_name or f"tad_{tad_index}_zoom.png"
        out = self.output_dir / save_name
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.show()

    def plot_tad_statistics(self, figsize=(14, 5),
                            save_name="tad_statistics.png"):
        sizes = [(e - s) * self.resolution / 1000
                 for s, e in self.tad_borders]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # histogram
        axes[0].hist(sizes, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.median(sizes), color='red', ls='--', lw=2,
                        label=f'Median: {np.median(sizes):.1f} kb')
        axes[0].set_xlabel('TAD Size (kb)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('TAD Size Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # boxplot
        axes[1].boxplot(sizes, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue'),
                        medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('TAD Size (kb)')
        axes[1].set_title('TAD Size Statistics', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        stats_text = (f'Mean: {np.mean(sizes):.1f} kb\n'
                      f'Median: {np.median(sizes):.1f} kb\n'
                      f'Std: {np.std(sizes):.1f} kb\n'
                      f'Min: {np.min(sizes):.1f} kb\n'
                      f'Max: {np.max(sizes):.1f} kb')
        axes[1].text(1.15, np.median(sizes), stats_text,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     fontsize=9)

        # coverage
        n_bins = self.contact_matrix.shape[0]
        cov = np.zeros(n_bins)
        for s, e in self.tad_borders:
            cov[s:e + 1] = 1
        pos_mb = np.arange(n_bins) * self.resolution / 1_000_000
        axes[2].fill_between(pos_mb, 0, cov, alpha=0.5, color='green')
        axes[2].set_xlabel('Genomic Position (Mb)')
        axes[2].set_ylabel('TAD Coverage')
        axes[2].set_title('TAD Coverage Along Chromosome', fontweight='bold')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(axis='y', alpha=0.3)
        pct = np.sum(cov) / n_bins * 100
        axes[2].text(0.02, 0.98,
                     f'Coverage: {pct:.1f}%\nTotal TADs: {len(self.tad_borders)}',
                     transform=axes[2].transAxes, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     fontsize=9)

        plt.tight_layout()
        out = self.output_dir / save_name
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.show()

    def plot_tad_heatmap_annotated(self, max_tads=20, figsize=(14, 12),
                                    cmap='YlOrRd',
                                    save_name="tad_heatmap_annotated.png"):
        fig, ax = plt.subplots(figsize=figsize)
        matrix = self.contact_matrix.copy()
        matrix[matrix == 0] = np.nan

        im = ax.imshow(matrix, cmap=cmap, norm=LogNorm(),
                       interpolation='none', origin='upper', aspect='auto')

        n_show = min(len(self.tad_borders), max_tads)
        for i, (s, e) in enumerate(self.tad_borders[:n_show]):
            self._draw_tad_rectangles(ax, [(s, e)], color='blue', linewidth=2)
            c = (s + e) / 2
            ax.text(c, c, str(i + 1), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.8))

        self._setup_axes(ax, matrix.shape[0])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Contact Frequency (log)', rotation=270, labelpad=20)
        ax.set_title(f'Numbered TADs (first {n_show} of {len(self.tad_borders)})',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        out = self.output_dir / save_name
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.show()

    def plot_diagonal_view(self, window_size=50, figsize=(14, 6),
                           save_name="diagonal_view.png"):
        n = self.contact_matrix.shape[0]
        masked = np.zeros_like(self.contact_matrix)
        for i in range(n):
            lo = max(0, i - window_size)
            hi = min(n, i + window_size + 1)
            masked[i, lo:hi] = self.contact_matrix[i, lo:hi]
        masked[masked == 0] = np.nan

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(masked, cmap='RdPu', norm=LogNorm(),
                       interpolation='none', origin='upper')
        self._draw_tad_rectangles(ax, self.tad_borders,
                                  color='cyan', linewidth=1.5, alpha=0.9)
        self._setup_axes(ax, n)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Diagonal View (±{window_size} bins) with TADs',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = self.output_dir / save_name
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.show()

    def plot_comparison_subplots(self, tad_indices=None, n_cols=3,
                                 figsize_per=(4, 4),
                                 save_name="tad_comparison.png"):
        if tad_indices is None:
            tad_indices = list(range(min(6, len(self.tad_borders))))

        n = len(tad_indices)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        ctx = 5
        for idx, ti in enumerate(tad_indices):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            if ti >= len(self.tad_borders):
                ax.axis('off')
                continue

            s, e = self.tad_borders[ti]
            s0 = max(0, s - ctx)
            e0 = min(self.contact_matrix.shape[0], e + ctx)
            sub = self.contact_matrix[s0:e0, s0:e0]

            ax.imshow(sub, cmap='YlOrRd', interpolation='none')
            rs, re = s - s0, e - s0
            rect = patches.Rectangle(
                (rs - 0.5, rs - 0.5), re - rs + 1, re - rs + 1,
                linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            kb = (e - s) * self.resolution / 1000
            ax.set_title(f'TAD #{ti + 1} ({kb:.0f} kb)', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(n, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].axis('off')

        plt.suptitle('TAD Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        out = self.output_dir / save_name
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.show()

    def generate_all_plots(self):
        """Generate the full suite of visualizations"""
        print("1/6 Contact map with TADs...")
        self.plot_contact_map_with_tads()
        print("2/6 TAD statistics...")
        self.plot_tad_statistics()
        print("3/6 Annotated heatmap...")
        self.plot_tad_heatmap_annotated()
        print("4/6 Diagonal view...")
        self.plot_diagonal_view()
        print("5/6 TAD zoom...")
        if self.tad_borders:
            self.plot_tad_zoom(0)
        print("6/6 TAD comparison subplots...")
        self.plot_comparison_subplots()
        print(f"\n✓ All visualizations saved to: {self.output_dir}/")


# ==============================================================
# Interactive mcool-based visualization (requires cooler)
# ==============================================================

def plot_mcool_with_tads(mcool_file: str, tad_file: str,
                         chromosome: str = "chr19",
                         resolution: int = 50000,
                         start: Optional[int] = None,
                         end: Optional[int] = None,
                         save: bool = True):
    """
    Visualize Hi-C map directly from .mcool with TAD overlays.

    Requires: cooler
    """
    import cooler
    from matplotlib.patches import Rectangle

    uri = f"{mcool_file}::resolutions/{resolution}"
    clr = cooler.Cooler(uri)

    if start and end:
        region = f"{chromosome}:{start}-{end}"
        matrix = clr.matrix(balance=True).fetch(region)
        start_bin = start // resolution
    else:
        matrix = clr.matrix(balance=True).fetch(chromosome)
        start_bin = 0

    matrix = np.nan_to_num(matrix, nan=0.0)

    tad_borders = np.loadtxt(tad_file, dtype=int)
    if tad_borders.ndim == 1:
        tad_borders = tad_borders.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.matshow(matrix, cmap='RdYlBu_r',
                    norm=plt.matplotlib.colors.LogNorm())

    for st, en in tad_borders:
        rs = st - start_bin
        re = en - start_bin
        if 0 <= rs < matrix.shape[0] and 0 <= re < matrix.shape[0]:
            rect = Rectangle((rs, rs), re - rs, re - rs,
                             fill=False, edgecolor='cyan', linewidth=2)
            ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label='Contact Frequency')
    ax.set_title(f'{chromosome} Hi-C with TADs', fontsize=14, pad=20)
    plt.tight_layout()

    if save:
        out = f'{chromosome}_tads_interactive.png'
        plt.savefig(out, dpi=300)
        print(f"Saved: {out}")
    plt.show()


# ==============================================================
# Convenience function
# ==============================================================

def visualize_coitad_results(results_dir: str, data_dir: str,
                             chromosome: str = "chr19",
                             resolution: int = 40000,
                             best_radius: int = 2,
                             algorithm: str = "HDBSCAN"):
    """One-call visualization after coiTAD run"""
    cm_file = f"{data_dir}/{chromosome}_{resolution // 1000}kb.hic"
    tad_file = f"{results_dir}/TADs/{algorithm}_{best_radius}_TAD_BinID.txt"

    print(f"Contact matrix : {cm_file}")
    print(f"TAD file       : {tad_file}")

    viz = CoiTADVisualizer(
        contact_matrix_file=cm_file,
        tad_file=tad_file,
        resolution=resolution,
        output_dir=f"{results_dir}/visualizations")
    viz.generate_all_plots()
    return viz