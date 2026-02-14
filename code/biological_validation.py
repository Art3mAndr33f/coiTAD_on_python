# biological_validation.py
"""
Biological validation of TAD boundaries using ChIP-Seq data
Биологическая валидация границ TAD с использованием ChIP-Seq данных
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
from io import StringIO
import warnings

warnings.filterwarnings('ignore')


class ChIPSeqDataLoader:
    """Load ChIP-Seq peak data from UCSC or custom files"""
    
    def __init__(self, genome: str = 'hg19'):
        self.genome = genome
        self.ucsc_base = "https://hgdownload.soe.ucsc.edu/goldenPath"
        
    def load_encode_peaks(self, track_name: str, chromosome: str) -> pd.DataFrame:
        """
        Load ENCODE ChIP-Seq peaks from UCSC
        
        Args:
            track_name: Name of the track (e.g., 'wgEncodeBroadHistoneH1hescCtcfStdPk')
            chromosome: Chromosome name
            
        Returns:
            DataFrame with columns: chr, start, end, name, score
        """
        print(f"Loading {track_name} for {chromosome}...")
        
        # Construct URL
        url = f"{self.ucsc_base}/{self.genome}/encodeDCC/wgEncodeBroadHistone/{track_name}.broadPeak.gz"
        
        try:
            # Download and parse
            df = pd.read_csv(url, sep='\t', compression='gzip', header=None,
                           names=['chr', 'start', 'end', 'name', 'score', 
                                 'strand', 'signalValue', 'pValue', 'qValue'])
            
            # Filter by chromosome
            df = df[df['chr'] == chromosome].copy()
            
            print(f"  Loaded {len(df)} peaks")
            return df[['chr', 'start', 'end', 'name', 'score']]
            
        except Exception as e:
            print(f"  Warning: Could not load from UCSC: {e}")
            return pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'score'])
    
    def load_custom_peaks(self, bed_file: str, chromosome: Optional[str] = None) -> pd.DataFrame:
        """
        Load peaks from custom BED file
        
        Args:
            bed_file: Path to BED file
            chromosome: Filter by chromosome (optional)
            
        Returns:
            DataFrame with peak data
        """
        print(f"Loading custom peaks from {bed_file}...")
        
        # Determine number of columns
        with open(bed_file, 'r') as f:
            first_line = f.readline().strip()
            n_cols = len(first_line.split('\t'))
        
        # Standard BED columns
        if n_cols >= 3:
            names = ['chr', 'start', 'end']
            if n_cols >= 4:
                names.append('name')
            if n_cols >= 5:
                names.append('score')
            
            df = pd.read_csv(bed_file, sep='\t', header=None, names=names[:n_cols])
            
            # Filter by chromosome if specified
            if chromosome:
                df = df[df['chr'] == chromosome].copy()
            
            print(f"  Loaded {len(df)} peaks")
            return df
        else:
            raise ValueError(f"Invalid BED file format: {bed_file}")


class BiologicalValidator:
    """Validate TAD boundaries using biological markers"""
    
    def __init__(self, 
                 resolution: int,
                 chromosome: str,
                 genome: str = 'hg19',
                 output_dir: str = 'biological_validation'):
        """
        Args:
            resolution: Hi-C resolution in bp
            chromosome: Chromosome name
            genome: Genome build
            output_dir: Output directory
        """
        self.resolution = resolution
        self.chromosome = chromosome
        self.genome = genome
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.chip_loader = ChIPSeqDataLoader(genome)
        self.markers = {}
        
    def load_marker_data(self, marker_files: Dict[str, str] = None):
        """
        Load ChIP-Seq data for various markers
        
        Args:
            marker_files: Dict mapping marker names to BED file paths
                         If None, attempts to load from UCSC ENCODE
        """
        if marker_files:
            # Load from custom files
            for marker, filepath in marker_files.items():
                self.markers[marker] = self.chip_loader.load_custom_peaks(
                    filepath, self.chromosome
                )
        else:
            # Try to load from UCSC ENCODE (hESC data)
            encode_tracks = {
                'CTCF': 'wgEncodeBroadHistoneH1hescCtcfStdPk',
                'H3K4me1': 'wgEncodeBroadHistoneH1hescH3k4me1StdPk',
                'H3K4me3': 'wgEncodeBroadHistoneH1hescH3k4me3StdPk',
                'H3K27ac': 'wgEncodeBroadHistoneH1hescH3k27acStdPk',
                'RNAPII': 'wgEncodeSydhTfbsH1hescPol2StdPk'
            }
            
            for marker, track in encode_tracks.items():
                df = self.chip_loader.load_encode_peaks(track, self.chromosome)
                if len(df) > 0:
                    self.markers[marker] = df
                else:
                    print(f"  Warning: No data loaded for {marker}")
    
    def peaks_to_bins(self, peaks_df: pd.DataFrame, n_bins: int) -> np.ndarray:
        """
        Convert peak positions to bin coverage
        
        Args:
            peaks_df: DataFrame with peak positions
            n_bins: Number of genomic bins
            
        Returns:
            Array of peak counts per bin
        """
        bin_counts = np.zeros(n_bins, dtype=int)
        
        for _, peak in peaks_df.iterrows():
            start_bin = int(peak['start'] / self.resolution)
            end_bin = int(peak['end'] / self.resolution)
            
            # Clip to valid range
            start_bin = max(0, min(start_bin, n_bins - 1))
            end_bin = max(0, min(end_bin, n_bins - 1))
            
            bin_counts[start_bin:end_bin + 1] += 1
        
        return bin_counts
    
    def calculate_boundary_enrichment(self, 
                                     tad_borders: np.ndarray,
                                     marker_name: str,
                                     n_bins: int,
                                     window: int = 10) -> Dict:
        """
        Calculate enrichment of marker around TAD boundaries
        
        Args:
            tad_borders: TAD boundaries [[start, end], ...]
            marker_name: Name of the marker
            n_bins: Total number of bins
            window: Window size around boundary (in bins)
            
        Returns:
            Dict with enrichment statistics
        """
        if marker_name not in self.markers:
            raise ValueError(f"Marker {marker_name} not loaded")
        
        # Convert peaks to bins
        bin_peaks = self.peaks_to_bins(self.markers[marker_name], n_bins)
        
        # Extract boundaries
        boundaries = []
        for start, end in tad_borders:
            boundaries.append(start)
            boundaries.append(end)
        boundaries = sorted(set(boundaries))
        
        # Calculate enrichment around each boundary
        enrichment_profile = np.zeros(2 * window + 1)
        valid_boundaries = 0
        
        for boundary in boundaries:
            if window <= boundary < n_bins - window:
                profile = bin_peaks[boundary - window:boundary + window + 1]
                enrichment_profile += profile
                valid_boundaries += 1
        
        # Normalize
        if valid_boundaries > 0:
            enrichment_profile /= valid_boundaries
        
        # Calculate peaks per bin at boundaries
        boundary_peaks = []
        for boundary in boundaries:
            if 0 <= boundary < n_bins:
                boundary_peaks.append(bin_peaks[boundary])
        
        avg_peaks_per_bin = np.mean(boundary_peaks) if boundary_peaks else 0
        
        return {
            'enrichment_profile': enrichment_profile,
            'avg_peaks_per_bin': avg_peaks_per_bin,
            'total_boundary_peaks': sum(boundary_peaks),
            'n_boundaries': len(boundaries),
            'positions': np.arange(-window, window + 1)
        }
    
    def validate_method(self, 
                       tad_borders: np.ndarray,
                       n_bins: int,
                       method_name: str) -> pd.DataFrame:
        """
        Validate TAD boundaries for a single method
        
        Args:
            tad_borders: TAD boundaries
            n_bins: Total number of bins
            method_name: Name of the method
            
        Returns:
            DataFrame with validation results
        """
        results = []
        
        for marker_name in self.markers.keys():
            enrichment = self.calculate_boundary_enrichment(
                tad_borders, marker_name, n_bins, window=10
            )
            
            results.append({
                'Method': method_name,
                'Marker': marker_name,
                'Avg_Peaks_Per_Bin': enrichment['avg_peaks_per_bin'],
                'Total_Boundary_Peaks': enrichment['total_boundary_peaks'],
                'N_Boundaries': enrichment['n_boundaries']
            })
        
        return pd.DataFrame(results)
    
    def compare_methods(self, 
                       method_tads: Dict[str, np.ndarray],
                       n_bins: int) -> pd.DataFrame:
        """
        Compare biological validation across multiple methods
        
        Args:
            method_tads: Dict mapping method names to TAD boundaries
            n_bins: Total number of bins
            
        Returns:
            Combined validation results
        """
        all_results = []
        
        for method_name, tad_borders in method_tads.items():
            print(f"\nValidating {method_name}...")
            results = self.validate_method(tad_borders, n_bins, method_name)
            all_results.append(results)
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Save results
        output_file = self.output_dir / 'validation_results.csv'
        combined.to_csv(output_file, index=False)
        print(f"\nValidation results saved: {output_file}")
        
        return combined
    
    def plot_enrichment_profiles(self,
                                method_tads: Dict[str, np.ndarray],
                                n_bins: int,
                                markers: List[str] = None,
                                window: int = 500):
        """
        Plot enrichment profiles for all methods and markers
        
        Args:
            method_tads: Dict mapping method names to TAD boundaries
            n_bins: Total number of bins
            markers: List of markers to plot (None = all)
            window: Window size in kb
        """
        if markers is None:
            markers = list(self.markers.keys())
        
        n_markers = len(markers)
        fig, axes = plt.subplots(1, n_markers, figsize=(5 * n_markers, 4))
        
        if n_markers == 1:
            axes = [axes]
        
        window_bins = int(window * 1000 / self.resolution)
        
        for idx, marker in enumerate(markers):
            ax = axes[idx]
            
            for method_name, tad_borders in method_tads.items():
                enrichment = self.calculate_boundary_enrichment(
                    tad_borders, marker, n_bins, window=window_bins
                )
                
                positions_kb = enrichment['positions'] * self.resolution / 1000
                
                ax.plot(positions_kb, enrichment['enrichment_profile'],
                       label=method_name, linewidth=2)
            
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Distance from TAD boundary (kb)', fontsize=11)
            ax.set_ylabel(f'Average {marker} peaks per bin', fontsize=11)
            ax.set_title(f'{marker} Enrichment', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'enrichment_profiles.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enrichment profiles saved: {output_file}")
    
    def plot_peaks_per_bin_comparison(self,
                                     method_tads: Dict[str, np.ndarray],
                                     n_bins: int):
        """
        Plot average peaks per bin comparison (like Figure 9c in paper)
        
        Args:
            method_tads: Dict mapping method names to TAD boundaries
            n_bins: Total number of bins
        """
        markers = list(self.markers.keys())
        n_markers = len(markers)
        
        fig, axes = plt.subplots(1, n_markers, figsize=(4 * n_markers, 4))
        
        if n_markers == 1:
            axes = [axes]
        
        for idx, marker in enumerate(markers):
            ax = axes[idx]
            
            methods = []
            peaks_per_bin = []
            
            for method_name, tad_borders in method_tads.items():
                enrichment = self.calculate_boundary_enrichment(
                    tad_borders, marker, n_bins, window=2
                )
                
                methods.append(method_name)
                peaks_per_bin.append(enrichment['avg_peaks_per_bin'])
            
            # Bar plot
            colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
            bars = ax.bar(methods, peaks_per_bin, color=colors, 
                         edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel(f'Average {marker} peaks per bin', fontsize=11)
            ax.set_title(f'{marker} at Boundaries', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = self.output_dir / 'peaks_per_bin_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Peaks per bin comparison saved: {output_file}")
    
    def generate_validation_report(self,
                                   validation_results: pd.DataFrame) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("BIOLOGICAL VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"Chromosome: {self.chromosome}")
        report.append(f"Resolution: {self.resolution / 1000:.0f} kb")
        report.append(f"Genome: {self.genome}")
        report.append("")
        
        report.append("ENRICHMENT SUMMARY (Average Peaks Per Bin at TAD Boundaries)")
        report.append("-" * 80)
        
        # Pivot table
        pivot = validation_results.pivot(index='Marker', columns='Method', 
                                        values='Avg_Peaks_Per_Bin')
        report.append(pivot.to_string())
        report.append("")
        
        # Find best method for each marker
        report.append("BEST PERFORMING METHOD FOR EACH MARKER")
        report.append("-" * 80)
        
        for marker in validation_results['Marker'].unique():
            marker_data = validation_results[validation_results['Marker'] == marker]
            best_method = marker_data.loc[marker_data['Avg_Peaks_Per_Bin'].idxmax(), 'Method']
            best_value = marker_data['Avg_Peaks_Per_Bin'].max()
            
            report.append(f"{marker:15s}: {best_method:15s} ({best_value:.3f} peaks/bin)")
        
        report.append("")
        
        # Overall ranking
        report.append("OVERALL METHOD RANKING (by average enrichment across all markers)")
        report.append("-" * 80)
        
        method_avg = validation_results.groupby('Method')['Avg_Peaks_Per_Bin'].mean().sort_values(ascending=False)
        
        for rank, (method, avg_enrichment) in enumerate(method_avg.items(), 1):
            report.append(f"{rank}. {method:15s}: {avg_enrichment:.3f} average peaks/bin")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.output_dir / 'validation_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nValidation report saved: {report_file}")
        print(report_text)
        
        return report_text