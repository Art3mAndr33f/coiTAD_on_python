"""
Biological validation of TAD boundaries using ChIP-Seq data.
Includes downloading helpers for ENCODE / UCSC data.
"""

import gzip
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

warnings.filterwarnings('ignore')


# ======================================================================
# ChIP-Seq data downloading
# ======================================================================

class ChIPSeqDownloader:
    """Download ChIP-Seq peak data from ENCODE / UCSC"""

    def __init__(self, output_dir: str = "chipseq_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def download_file(self, url: str, save_as: str) -> bool:
        """
        Download file from URL, decompress .gz if needed,
        save with the specified name.
        """
        try:
            print(f"Downloading {url} ...")
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            is_gz = url.endswith('.gz')

            if is_gz:
                tmp_path = self.output_dir / (save_as + '.gz')
                with open(tmp_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                print("  Decompressing...")
                final_path = self.output_dir / save_as
                with gzip.open(tmp_path, 'rb') as f_in:
                    with open(final_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                tmp_path.unlink()
            else:
                final_path = self.output_dir / save_as
                with open(final_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"  Saved to {final_path}")
            return True

        except Exception as e:
            print(f"  Error: {e}")
            return False

    def download_hesc_markers(self, genome: str = 'hg19') -> Dict[str, str]:
        """Download hESC histone / TF peak files from UCSC ENCODE"""
        base_hist = (
            f"https://hgdownload.soe.ucsc.edu/goldenPath/"
            f"{genome}/encodeDCC/wgEncodeBroadHistone"
        )
        base_tfbs = (
            f"https://hgdownload.soe.ucsc.edu/goldenPath/"
            f"{genome}/encodeDCC/wgEncodeSydhTfbs"
        )

        tracks = {
            'CTCF': f"{base_hist}/wgEncodeBroadHistoneH1hescCtcfStdPk.broadPeak.gz",
            'H3K4me1': f"{base_hist}/wgEncodeBroadHistoneH1hescH3k4me1StdPk.broadPeak.gz",
            'H3K4me3': f"{base_hist}/wgEncodeBroadHistoneH1hescH3k4me3StdPk.broadPeak.gz",
            'H3K27ac': f"{base_hist}/wgEncodeBroadHistoneH1hescH3k27acStdPk.broadPeak.gz",
        }

        rnapii_urls = [
            f"{base_tfbs}/wgEncodeSydhTfbsH1hescPol2StdPk.narrowPeak.gz",
            f"{base_tfbs}/wgEncodeSydhTfbsH1hescPol2IggmusPk.narrowPeak.gz",
            f"{base_hist}/wgEncodeBroadHistoneH1hescPol2bStdPk.broadPeak.gz",
        ]

        downloaded: Dict[str, str] = {}

        # Download histone marks
        for marker, url in tracks.items():
            local_name = f"{marker}_{genome}.bed"
            if self.download_file(url, local_name):
                downloaded[marker] = str(self.output_dir / local_name)

        # Try RNAPII alternatives
        print("\nAttempting to download RNA Pol II data...")
        rnapii_local = f"RNAPII_{genome}.bed"
        for url in rnapii_urls:
            if self.download_file(url, rnapii_local):
                downloaded['RNAPII'] = str(self.output_dir / rnapii_local)
                break
        else:
            print("  Warning: could not download RNAPII data (skipping)")

        return downloaded


def prepare_chipseq_data(genome: str = 'hg19',
                         output_dir: str = 'chipseq_data') -> Dict[str, str]:
    """Convenience wrapper: download all hESC markers"""
    dl = ChIPSeqDownloader(output_dir)
    marker_files = dl.download_hesc_markers(genome)
    print(f"\nDownloaded {len(marker_files)} marker datasets:")
    for m, fp in marker_files.items():
        print(f"  {m}: {fp}")
    return marker_files


# ======================================================================
# ChIP-Seq peak loader
# ======================================================================

class ChIPSeqDataLoader:
    """Load ChIP-Seq peaks from local BED or remote UCSC files"""

    def __init__(self, genome: str = 'hg19'):
        self.genome = genome

    def load_custom_peaks(self, bed_file: str,
                          chromosome: Optional[str] = None) -> pd.DataFrame:
        """Load peaks from a local BED / broadPeak / narrowPeak file"""
        print(f"Loading peaks from {bed_file}...")

        # Detect column count
        with open(bed_file, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                print("  Warning: empty file")
                return pd.DataFrame(columns=['chr', 'start', 'end'])
            n_cols = len(first_line.split('\t'))

        # broadPeak has 9 columns, narrowPeak has 10, BED has 3-6
        all_names = ['chr', 'start', 'end', 'name', 'score',
                     'strand', 'signalValue', 'pValue', 'qValue', 'peak']
        use_names = all_names[:min(n_cols, len(all_names))]

        df = pd.read_csv(bed_file, sep='\t', header=None,
                         names=use_names, comment='#')

        if chromosome and 'chr' in df.columns:
            df = df[df['chr'] == chromosome].copy()

        print(f"  Loaded {len(df)} peaks")
        return df


# ======================================================================
# Biological validator
# ======================================================================

class BiologicalValidator:
    """Validate TAD boundaries against ChIP-Seq markers"""

    def __init__(self,
                 resolution: int,
                 chromosome: str,
                 genome: str = 'hg19',
                 output_dir: str = 'biological_validation'):
        self.resolution = resolution
        self.chromosome = chromosome
        self.genome = genome
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.chip_loader = ChIPSeqDataLoader(genome)
        self.markers: Dict[str, pd.DataFrame] = {}

    # ---------- data loading ----------

    def load_marker_data(self, marker_files: Optional[Dict[str, str]] = None):
        """Load ChIP-Seq peaks (custom BED files or UCSC ENCODE)"""
        if marker_files:
            for marker, fp in marker_files.items():
                try:
                    df = self.chip_loader.load_custom_peaks(fp, self.chromosome)
                    if len(df) > 0:
                        self.markers[marker] = df
                    else:
                        print(f"  Warning: no peaks for {marker} on {self.chromosome}")
                except Exception as e:
                    print(f"  Warning: failed to load {marker}: {e}")
        else:
            # Fallback: try UCSC direct download (broadPeak)
            base = (f"https://hgdownload.soe.ucsc.edu/goldenPath/"
                    f"{self.genome}/encodeDCC/wgEncodeBroadHistone")
            encode_tracks = {
                'CTCF': 'wgEncodeBroadHistoneH1hescCtcfStdPk',
                'H3K4me1': 'wgEncodeBroadHistoneH1hescH3k4me1StdPk',
                'H3K4me3': 'wgEncodeBroadHistoneH1hescH3k4me3StdPk',
                'H3K27ac': 'wgEncodeBroadHistoneH1hescH3k27acStdPk',
            }
            for marker, track in encode_tracks.items():
                url = f"{base}/{track}.broadPeak.gz"
                print(f"Loading {marker} from UCSC...")
                try:
                    df = pd.read_csv(
                        url, sep='\t', compression='gzip', header=None,
                        names=['chr', 'start', 'end', 'name', 'score',
                               'strand', 'signalValue', 'pValue', 'qValue'])
                    df = df[df['chr'] == self.chromosome].copy()
                    if len(df) > 0:
                        self.markers[marker] = df
                        print(f"  Loaded {len(df)} peaks")
                    else:
                        print(f"  No peaks on {self.chromosome}")
                except Exception as e:
                    print(f"  Warning: {e}")

        print(f"\nLoaded markers: {list(self.markers.keys())}")

    # ---------- helpers ----------

    def peaks_to_bins(self, peaks_df: pd.DataFrame,
                      n_bins: int) -> np.ndarray:
        """Convert peak positions -> per-bin counts"""
        counts = np.zeros(n_bins, dtype=int)
        for _, pk in peaks_df.iterrows():
            sb = max(0, min(int(pk['start'] / self.resolution), n_bins - 1))
            eb = max(0, min(int(pk['end'] / self.resolution), n_bins - 1))
            counts[sb:eb + 1] += 1
        return counts

    # ---------- enrichment ----------

    def calculate_boundary_enrichment(
            self,
            tad_borders: np.ndarray,
            marker_name: str,
            n_bins: int,
            window: int = 10
    ) -> Dict:
        """Average marker profile around TAD boundaries"""
        if marker_name not in self.markers:
            raise ValueError(f"Marker {marker_name} not loaded")

        bin_peaks = self.peaks_to_bins(self.markers[marker_name], n_bins)

        boundaries = sorted({int(b) for tad in tad_borders for b in tad})

        profile = np.zeros(2 * window + 1)
        valid = 0
        for b in boundaries:
            if window <= b < n_bins - window:
                profile += bin_peaks[b - window:b + window + 1]
                valid += 1
        if valid > 0:
            profile /= valid

        boundary_peaks = [bin_peaks[b] for b in boundaries if 0 <= b < n_bins]
        avg = float(np.mean(boundary_peaks)) if boundary_peaks else 0.0

        return {
            'enrichment_profile': profile,
            'avg_peaks_per_bin': avg,
            'total_boundary_peaks': sum(boundary_peaks),
            'n_boundaries': len(boundaries),
            'positions': np.arange(-window, window + 1),
        }

    # ---------- per-method / multi-method ----------

    def validate_method(self, tad_borders: np.ndarray,
                        n_bins: int,
                        method_name: str) -> pd.DataFrame:
        rows = []
        for marker in self.markers:
            e = self.calculate_boundary_enrichment(tad_borders, marker, n_bins)
            rows.append({
                'Method': method_name,
                'Marker': marker,
                'Avg_Peaks_Per_Bin': e['avg_peaks_per_bin'],
                'Total_Boundary_Peaks': e['total_boundary_peaks'],
                'N_Boundaries': e['n_boundaries'],
            })
        return pd.DataFrame(rows)

    def compare_methods(self, method_tads: Dict[str, np.ndarray],
                        n_bins: int) -> pd.DataFrame:
        parts = []
        for name, borders in method_tads.items():
            print(f"\nValidating {name}...")
            parts.append(self.validate_method(borders, n_bins, name))

        combined = pd.concat(parts, ignore_index=True)
        out = self.output_dir / 'validation_results.csv'
        combined.to_csv(out, index=False)
        print(f"\nValidation results saved: {out}")
        return combined

    # ---------- plots ----------

    def plot_enrichment_profiles(self,
                                method_tads: Dict[str, np.ndarray],
                                n_bins: int,
                                markers: Optional[List[str]] = None,
                                window: int = 500):
        markers = markers or list(self.markers.keys())
        n_markers = len(markers)
        if n_markers == 0:
            print("No markers to plot")
            return

        fig, axes = plt.subplots(1, n_markers, figsize=(5 * n_markers, 4))
        if n_markers == 1:
            axes = [axes]

        window_bins = int(window * 1000 / self.resolution)

        for idx, marker in enumerate(markers):
            ax = axes[idx]
            for method, borders in method_tads.items():
                e = self.calculate_boundary_enrichment(
                    borders, marker, n_bins, window=window_bins)
                pos_kb = e['positions'] * self.resolution / 1000
                ax.plot(pos_kb, e['enrichment_profile'], label=method, linewidth=2)
            ax.axvline(0, color='black', ls='--', lw=1, alpha=0.5)
            ax.set_xlabel('Distance from boundary (kb)')
            ax.set_ylabel(f'Avg {marker} peaks/bin')
            ax.set_title(f'{marker} Enrichment', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.output_dir / 'enrichment_profiles.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enrichment profiles saved: {out}")

    def plot_peaks_per_bin_comparison(self,
                                     method_tads: Dict[str, np.ndarray],
                                     n_bins: int):
        markers = list(self.markers.keys())
        n_markers = len(markers)
        if n_markers == 0:
            print("No markers to plot")
            return

        fig, axes = plt.subplots(1, n_markers, figsize=(4 * n_markers, 4))
        if n_markers == 1:
            axes = [axes]

        for idx, marker in enumerate(markers):
            ax = axes[idx]
            methods, vals = [], []
            for method, borders in method_tads.items():
                e = self.calculate_boundary_enrichment(
                    borders, marker, n_bins, window=2)
                methods.append(method)
                vals.append(e['avg_peaks_per_bin'])

            colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
            bars = ax.bar(methods, vals, color=colors,
                          edgecolor='black', linewidth=1.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{h:.2f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
            ax.set_ylabel(f'Avg {marker} peaks/bin')
            ax.set_title(f'{marker} at Boundaries', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        out = self.output_dir / 'peaks_per_bin_comparison.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Peaks per bin comparison saved: {out}")

    # ---------- report ----------

    def generate_validation_report(self,
                                   validation_results: pd.DataFrame) -> str:
        lines = [
            "=" * 80,
            "BIOLOGICAL VALIDATION REPORT",
            "=" * 80, "",
            f"Chromosome: {self.chromosome}",
            f"Resolution: {self.resolution / 1000:.0f} kb",
            f"Genome:     {self.genome}", "",
            "ENRICHMENT SUMMARY (Avg Peaks Per Bin at TAD Boundaries)",
            "-" * 80,
        ]

        pivot = validation_results.pivot(
            index='Marker', columns='Method', values='Avg_Peaks_Per_Bin')
        lines.append(pivot.to_string())
        lines.append("")

        lines.append("BEST METHOD PER MARKER")
        lines.append("-" * 80)
        for marker in validation_results['Marker'].unique():
            sub = validation_results[validation_results['Marker'] == marker]
            best_row = sub.loc[sub['Avg_Peaks_Per_Bin'].idxmax()]
            lines.append(
                f"  {marker:15s}: {best_row['Method']:15s} "
                f"({best_row['Avg_Peaks_Per_Bin']:.3f} peaks/bin)")
        lines.append("")

        lines.append("OVERALL RANKING (avg enrichment across markers)")
        lines.append("-" * 80)
        ranking = (validation_results.groupby('Method')['Avg_Peaks_Per_Bin']
                   .mean().sort_values(ascending=False))
        for rank, (method, val) in enumerate(ranking.items(), 1):
            lines.append(f"  {rank}. {method:15s}: {val:.3f} avg peaks/bin")
        lines.append("")
        lines.append("=" * 80)

        text = "\n".join(lines)
        out = self.output_dir / 'validation_report.txt'
        out.write_text(text)
        print(f"\nValidation report saved: {out}")
        print(text)
        return text