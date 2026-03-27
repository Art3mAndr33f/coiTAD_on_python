"""TAD comparison: MoC, ARI, NMI, Precision/Recall/F1, size stats, plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class TADComparator:
    def __init__(self, output_dir: str = "comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def load_tad_results(tad_file: str) -> np.ndarray:
        try:
            data = np.loadtxt(tad_file, dtype=int)
        except ValueError:
            data = np.loadtxt(tad_file, skiprows=1, usecols=(0, 2), dtype=int)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    @staticmethod
    def tads_to_labels(tads, n_bins):
        labels = np.zeros(n_bins, dtype=int)
        for tid, (s, e) in enumerate(tads, 1):
            labels[s:e+1] = tid
        return labels

    def calculate_moc(self, src, tgt, tolerance=0):
        if len(src) == 0 or len(tgt) == 0:
            return 0.0
        matched = 0
        for s in src:
            for t in tgt:
                if abs(s[0]-t[0]) <= tolerance and abs(s[1]-t[1]) <= tolerance:
                    matched += 1
                    break
        return matched / max(len(src), len(tgt))

    def calculate_clustering_metrics(self, t1, t2, n_bins):
        l1 = self.tads_to_labels(t1, n_bins)
        l2 = self.tads_to_labels(t2, n_bins)
        return {
            'Adjusted Rand Index': adjusted_rand_score(l1, l2),
            'Normalized Mutual Information': normalized_mutual_info_score(l1, l2),
        }

    def calculate_boundary_precision_recall(self, pred, true, tolerance=2):
        pb = set()
        for t in pred: pb.update([t[0], t[1]])
        tb = set()
        for t in true: tb.update([t[0], t[1]])
        tp = sum(1 for p in pb if any(abs(p-t) <= tolerance for t in tb))
        prec = tp / len(pb) if pb else 0
        rec = tp / len(tb) if tb else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        return {'Precision': prec, 'Recall': rec, 'F1-Score': f1}

    def compare_tad_sizes(self, method_results, resolution):
        stats = []
        for m, t in method_results.items():
            if len(t) == 0: continue
            s = (t[:,1]-t[:,0])*resolution/1000
            stats.append({'Method':m,'Count':len(t),'Mean (kb)':np.mean(s),
                          'Median (kb)':np.median(s),'Std (kb)':np.std(s),
                          'Min (kb)':np.min(s),'Max (kb)':np.max(s),
                          'Q1 (kb)':np.percentile(s,25),'Q3 (kb)':np.percentile(s,75)})
        return pd.DataFrame(stats)

    def plot_tad_size_comparison(self, method_results, resolution,
                                save_name="tad_size_comparison.png"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        rows = []
        for m, t in method_results.items():
            if len(t) > 0:
                for s in (t[:,1]-t[:,0])*resolution/1000:
                    rows.append({'Method':m,'TAD Size (kb)':s})
        df = pd.DataFrame(rows)
        sns.boxplot(data=df, x='Method', y='TAD Size (kb)', ax=axes[0])
        axes[0].set_title('TAD Size Distribution', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for m, t in method_results.items():
            if len(t) > 0:
                axes[1].hist((t[:,1]-t[:,0])*resolution/1000, bins=20,
                             alpha=0.5, label=m, edgecolor='black')
        axes[1].set_xlabel('TAD Size (kb)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('TAD Size Histogram', fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_tad_count_comparison(self, method_results,
                                  save_name="tad_count_comparison.png"):
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = list(method_results.keys())
        counts = [len(t) for t in method_results.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        bars = ax.bar(methods, counts, color=colors, edgecolor='black', linewidth=1.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2., b.get_height(),
                    f'{int(b.get_height())}', ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('Number of TADs')
        ax.set_title('TAD Count Comparison', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_moc_heatmap(self, method_results, save_name="moc_heatmap.png"):
        methods = list(method_results.keys())
        n = len(methods)
        mat = np.zeros((n, n))
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                mat[i,j] = self.calculate_moc(method_results[m1], method_results[m2], tolerance=2)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
        ax.set_xticklabels(methods); ax.set_yticklabels(methods)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center')
        ax.set_title('MoC (tolerance=2 bins)', fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        return mat, methods

    def generate_comparison_report(self, method_results, resolution, n_bins=None):
        lines = ["="*70, "TAD CALLING METHODS COMPARISON REPORT", "="*70, "",
                  "1. TAD COUNTS", "-"*70]
        for m, t in method_results.items():
            lines.append(f"  {m:20s}: {len(t):5d} TADs")
        lines += ["", "2. TAD SIZE STATISTICS", "-"*70,
                   self.compare_tad_sizes(method_results, resolution).to_string(index=False), ""]
        methods = list(method_results.keys())
        lines += ["3. MoC MATRIX [tolerance=2]", "-"*70]
        h = "            " + "  ".join(f"{m:>12s}" for m in methods)
        lines.append(h)
        for m1 in methods:
            row = f"  {m1:10s}"
            for m2 in methods:
                row += f"  {self.calculate_moc(method_results[m1], method_results[m2], 2):12.4f}"
            lines.append(row)
        lines.append("")
        if n_bins and len(methods) >= 2:
            lines += ["4. CLUSTERING METRICS", "-"*70]
            for i in range(len(methods)):
                for j in range(i+1, len(methods)):
                    cm = self.calculate_clustering_metrics(method_results[methods[i]], method_results[methods[j]], n_bins)
                    lines.append(f"  {methods[i]} vs {methods[j]}:")
                    for k, v in cm.items():
                        lines.append(f"    {k}: {v:.4f}")
        lines += ["", "="*70]
        text = "\n".join(lines)
        f = self.output_dir / "comparison_report.txt"
        f.write_text(text, encoding='utf-8')
        print(text)
        return text