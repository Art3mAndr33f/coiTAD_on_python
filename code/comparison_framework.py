"""
Framework for comparing HDBSCAN and OPTICS implementations
Инструменты для сравнения методов кластеризации
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr


class TADComparator:
    """Compare TAD calling results from different methods"""
    
    def __init__(self, output_dir: str = "comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_tad_results(self, tad_file: str) -> np.ndarray:
        """Load TAD boundaries from file"""
        try:
            data = np.loadtxt(tad_file, dtype=int)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data
        except:
            data = np.loadtxt(tad_file, skiprows=1, usecols=(0, 2), dtype=int)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data
    
    def calculate_moc(self, source_tads: np.ndarray, target_tads: np.ndarray, 
                     tolerance: int = 0) -> float:
        """
        Calculate Measure of Concordance (MoC)
        
        Args:
            source_tads: Source TAD boundaries [[start, end], ...]
            target_tads: Target TAD boundaries
            tolerance: Tolerance in bins for boundary matching
            
        Returns:
            MoC score (0 to 1)
        """
        if len(source_tads) == 0 or len(target_tads) == 0:
            return 0.0
        
        matched = 0
        
        for src_tad in source_tads:
            src_start, src_end = src_tad[0], src_tad[1]
            
            for tgt_tad in target_tads:
                tgt_start, tgt_end = tgt_tad[0], tgt_tad[1]
                
                # Check if boundaries match within tolerance
                start_match = abs(src_start - tgt_start) <= tolerance
                end_match = abs(src_end - tgt_end) <= tolerance
                
                if start_match and end_match:
                    matched += 1
                    break
        
        moc = matched / max(len(source_tads), len(target_tads))
        return moc
    
    def compare_tad_sizes(self, method_results: Dict[str, np.ndarray], 
                         resolution: int) -> pd.DataFrame:
        """
        Compare TAD size distributions across methods
        
        Returns:
            DataFrame with statistical summary
        """
        stats = []
        
        for method, tads in method_results.items():
            if len(tads) == 0:
                continue
                
            sizes = (tads[:, 1] - tads[:, 0]) * resolution / 1000  # in kb
            
            stats.append({
                'Method': method,
                'Count': len(tads),
                'Mean (kb)': np.mean(sizes),
                'Median (kb)': np.median(sizes),
                'Std (kb)': np.std(sizes),
                'Min (kb)': np.min(sizes),
                'Max (kb)': np.max(sizes),
                'Q1 (kb)': np.percentile(sizes, 25),
                'Q3 (kb)': np.percentile(sizes, 75)
            })
        
        return pd.DataFrame(stats)
    
    def plot_tad_size_comparison(self, method_results: Dict[str, np.ndarray],
                                resolution: int, save_name: str = "tad_size_comparison.png"):
        """Plot TAD size distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prepare data
        sizes_data = []
        methods = []
        
        for method, tads in method_results.items():
            if len(tads) > 0:
                sizes = (tads[:, 1] - tads[:, 0]) * resolution / 1000
                sizes_data.extend(sizes)
                methods.extend([method] * len(sizes))
        
        df = pd.DataFrame({'Method': methods, 'TAD Size (kb)': sizes_data})
        
        # Box plot
        sns.boxplot(data=df, x='Method', y='TAD Size (kb)', ax=axes[0])
        axes[0].set_title('TAD Size Distribution Comparison', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('TAD Size (kb)', fontsize=11)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Histogram
        for method in method_results.keys():
            if len(method_results[method]) > 0:
                sizes = (method_results[method][:, 1] - method_results[method][:, 0]) * resolution / 1000
                axes[1].hist(sizes, bins=20, alpha=0.5, label=method, edgecolor='black')
        
        axes[1].set_xlabel('TAD Size (kb)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('TAD Size Histogram', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_dir / save_name}")
    
    def plot_tad_count_comparison(self, method_results: Dict[str, np.ndarray],
                                 save_name: str = "tad_count_comparison.png"):
        """Plot TAD counts"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(method_results.keys())
        counts = [len(tads) for tads in method_results.values()]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        bars = ax.bar(methods, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
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
        """Plot MoC heatmap between all methods"""
        methods = list(method_results.keys())
        n_methods = len(methods)
        
        moc_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                moc = self.calculate_moc(
                    method_results[method1],
                    method_results[method2],
                    tolerance=2
                )
                moc_matrix[i, j] = moc
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(moc_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(n_methods))
        ax.set_yticks(np.arange(n_methods))
        ax.set_xticklabels(methods)
        ax.set_yticklabels(methods)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                text = ax.text(j, i, f'{moc_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title("Measure of Concordance (MoC) - Tolerance=2 bins",
                    fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('MoC Score', rotation=270, labelpad=20, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {self.output_dir / save_name}")
        
        return moc_matrix, methods
    
    def generate_comparison_report(self, method_results: Dict[str, np.ndarray],
                                  resolution: int) -> str:
        """Generate comprehensive comparison report"""
        report = []
        report.append("=" * 70)
        report.append("TAD CALLING METHODS COMPARISON REPORT")
        report.append("=" * 70)
        report.append("")
        
        # TAD counts
        report.append("1. TAD COUNTS")
        report.append("-" * 70)
        for method, tads in method_results.items():
            report.append(f"{method:20s}: {len(tads):5d} TADs")
        report.append("")
        
        # Size statistics
        report.append("2. TAD SIZE STATISTICS")
        report.append("-" * 70)
        stats_df = self.compare_tad_sizes(method_results, resolution)
        report.append(stats_df.to_string(index=False))
        report.append("")
        
        # MoC matrix
        report.append("3. MEASURE OF CONCORDANCE (MoC) MATRIX")
        report.append("-" * 70)
        methods = list(method_results.keys())
        
        # Header
        header = "          " + "  ".join([f"{m:10s}" for m in methods])
        report.append(header)
        
        for i, method1 in enumerate(methods):
            row = f"{method1:10s}"
            for method2 in methods:
                moc = self.calculate_moc(
                    method_results[method1],
                    method_results[method2],
                    tolerance=2
                )
                row += f"  {moc:10.4f}"
            report.append(row)
        
        report.append("")
        report.append("=" * 70)
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.output_dir / "comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nComparison report saved: {report_file}")
        print(report_text)
        
        return report_text