"""
TAD comparison framework with structural and clustering metrics.
Сравнение методов: MoC, размеры TAD, ARI, NMI, Precision/Recall/F1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class TADComparator:
    """Compare TAD calling results from different methods"""

    def __init__(self, output_dir: str = "comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_tad_results(tad_file: str) -> np.ndarray:
        """Load TAD boundaries from file"""
        try:
            data = np.loadtxt(tad_file, dtype=int)
        except ValueError:
            data = np.loadtxt(tad_file, skiprows=1, usecols=(0, 2), dtype=int)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    # ------------------------------------------------------------------
    # Label conversion (for clustering metrics)
    # ------------------------------------------------------------------

    @staticmethod
    def tads_to_labels(tads: np.ndarray, n_bins: int) -> np.ndarray:
        """Convert TAD boundaries to per-bin labels"""
        labels = np.zeros(n_bins, dtype=int)
        for tad_id, (start, end) in enumerate(tads, start=1):
            labels[start:end + 1] = tad_id
        return labels

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_moc(self, source_tads: np.ndarray, target_tads: np.ndarray,
                      tolerance: int = 0) -> float:
        """
        Measure of Concordance (MoC).

        Args:
            source_tads: [[start, end], ...]
            target_tads: [[start, end], ...]
            tolerance: tolerance in bins for boundary matching

        Returns:
            MoC score 0..1
        """
        if len(source_tads) == 0 or len(target_tads) == 0:
            return 0.0

        matched = 0
        for src in source_tads:
            for tgt in target_tads:
                if (abs(src[0] - tgt[0]) <= tolerance and
                        abs(src[1] - tgt[1]) <= tolerance):
                    matched += 1
                    break

        return matched / max(len(source_tads), len(target_tads))

    def calculate_clustering_metrics(self, tads1: np.ndarray, tads2: np.ndarray,
                                     n_bins: int) -> Dict[str, float]:
        """Adjusted Rand Index & Normalized Mutual Information"""
        labels1 = self.tads_to_labels(tads1, n_bins)
        labels2 = self.tads_to_labels(tads2, n_bins)
        return {
            'Adjusted Rand Index': adjusted_rand_score(labels1, labels2),
            'Normalized Mutual Information': normalized_mutual_info_score(labels1, labels2),
        }

    def calculate_boundary_precision_recall(
            self,
            pred_tads: np.ndarray,
            true_tads: np.ndarray,
            tolerance: int = 2
    ) -> Dict[str, float]:
        """Precision / Recall / F1 for boundary detection"""
        pred_boundaries = set()
        for t in pred_tads:
            pred_boundaries.update([t[0], t[1]])

        true_boundaries = set()
        for t in true_tads:
            true_boundaries.update([t[0], t[1]])

        tp = 0
        for pb in pred_boundaries:
            for tb in true_boundaries:
                if abs(pb - tb) <= tolerance:
                    tp += 1
                    break

        precision = tp / len(pred_boundaries) if pred_boundaries else 0
        recall = tp / len(true_boundaries) if true_boundaries else 0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0)

        return {'Precision': precision, 'Recall': recall, 'F1-Score': f1}

    # ------------------------------------------------------------------
    # Size statistics
    # ------------------------------------------------------------------

    def compare_tad_sizes(self, method_results: Dict[str, np.ndarray],
                          resolution: int) -> pd.DataFrame:
        """Statistical summary of TAD sizes across methods"""
        stats = []
        for method, tads in method_results.items():
            if len(tads) == 0:
                continue
            sizes = (tads[:, 1] - tads[:, 0]) * resolution / 1000  # kb
            stats.append({
                'Method': method,
                'Count': len(tads),
                'Mean (kb)': np.mean(sizes),
                'Median (kb)': np.median(sizes),
                'Std (kb)': np.std(sizes),
                'Min (kb)': np.min(sizes),
                'Max (kb)': np.max(sizes),
                'Q1 (kb)': np.percentile(sizes, 25),
                'Q3 (kb)': np.percentile(sizes, 75),
            })
        return pd.DataFrame(stats)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_tad_size_comparison(self, method_results: Dict[str, np.ndarray],
                                resolution: int,
                                save_name: str = "tad_size_comparison.png"):
        """Box-plot + histogram of TAD sizes"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        rows = []
        for method, tads in method_results.items():
            if len(tads) > 0:
                sizes = (tads[:, 1] - tads[:, 0]) * resolution / 1000
                rows.extend([{'Method': method, 'TAD Size (kb)': s} for s in sizes])

        df = pd.DataFrame(rows)

        sns.boxplot(data=df, x='Method', y='TAD Size (kb)', ax=axes[0])
        axes[0].set_title('TAD Size Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        for method, tads in method_results.items():
            if len(tads) > 0:
                sizes = (tads[:, 1] - tads[:, 0]) * resolution / 1000
                axes[1].hist(sizes, bins=20, alpha=0.5, label=method, edgecolor='black')
        axes[1].set_xlabel('TAD Size (kb)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('TAD Size Histogram', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / save_name}")

    def plot_tad_count_comparison(self, method_results: Dict[str, np.ndarray],
                                  save_name: str = "tad_count_comparison.png"):
        """Bar chart of TAD counts"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(method_results.keys())
        counts = [len(t) for t in method_results.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        bars = ax.bar(methods, counts, color=colors, edgecolor='black', linewidth=1.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f'{int(h)}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Number of TADs', fontsize=12)
        ax.set_title('TAD Count Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / save_name}")

    def plot_moc_heatmap(self, method_results: Dict[str, np.ndarray],
                         save_name: str = "moc_heatmap.png"):
        """MoC heatmap between all methods"""
        methods = list(method_results.keys())
        n = len(methods)
        mat = np.zeros((n, n))

        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                mat[i, j] = self.calculate_moc(
                    method_results[m1], method_results[m2], tolerance=2)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(methods)
        ax.set_yticklabels(methods)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                        color='black', fontsize=10)

        ax.set_title('Measure of Concordance (MoC) — Tolerance=2 bins',
                      fontsize=14, fontweight='bold', pad=20)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('MoC Score', rotation=270, labelpad=20, fontsize=11)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir / save_name}")
        return mat, methods

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_comparison_report(self, method_results: Dict[str, np.ndarray],
                                   resolution: int,
                                   n_bins: Optional[int] = None) -> str:
        """Comprehensive text report"""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("TAD CALLING METHODS COMPARISON REPORT")
        lines.append("=" * 70)
        lines.append("")

        # 1. Counts
        lines.append("1. TAD COUNTS")
        lines.append("-" * 70)
        for m, t in method_results.items():
            lines.append(f"  {m:20s}: {len(t):5d} TADs")
        lines.append("")

        # 2. Sizes
        lines.append("2. TAD SIZE STATISTICS")
        lines.append("-" * 70)
        lines.append(self.compare_tad_sizes(method_results, resolution).to_string(index=False))
        lines.append("")

        # 3. MoC
        lines.append("3. MEASURE OF CONCORDANCE (MoC) MATRIX  [tolerance=2]")
        lines.append("-" * 70)
        methods = list(method_results.keys())
        header = "            " + "  ".join(f"{m:>12s}" for m in methods)
        lines.append(header)
        for m1 in methods:
            row = f"  {m1:10s}"
            for m2 in methods:
                moc = self.calculate_moc(method_results[m1], method_results[m2], tolerance=2)
                row += f"  {moc:12.4f}"
            lines.append(row)
        lines.append("")

        # 4. Clustering metrics (if n_bins given)
        if n_bins and len(methods) >= 2:
            lines.append("4. CLUSTERING METRICS (pairwise)")
            lines.append("-" * 70)
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    cm = self.calculate_clustering_metrics(
                        method_results[methods[i]],
                        method_results[methods[j]],
                        n_bins)
                    lines.append(f"  {methods[i]} vs {methods[j]}:")
                    for k, v in cm.items():
                        lines.append(f"    {k}: {v:.4f}")
            lines.append("")

        lines.append("=" * 70)

        report_text = "\n".join(lines)
        report_file = self.output_dir / "comparison_report.txt"
        report_file.write_text(report_text)
        print(f"\nComparison report saved: {report_file}")
        print(report_text)
        return report_text