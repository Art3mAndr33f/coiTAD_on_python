"""Biological validation + ChIP-Seq data downloading."""

import gzip, shutil, warnings
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np, pandas as pd, matplotlib.pyplot as plt, requests

warnings.filterwarnings('ignore')


class ChIPSeqDownloader:
    def __init__(self, output_dir="chipseq_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def download_file(self, url, save_as):
        try:
            print(f"Downloading {url} ...")
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            if url.endswith('.gz'):
                tmp = self.output_dir / (save_as + '.gz')
                with open(tmp, 'wb') as f:
                    for ch in resp.iter_content(8192): f.write(ch)
                print("  Decompressing...")
                final = self.output_dir / save_as
                with gzip.open(tmp, 'rb') as fi, open(final, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
                tmp.unlink()
            else:
                final = self.output_dir / save_as
                with open(final, 'wb') as f:
                    for ch in resp.iter_content(8192): f.write(ch)
            print(f"  Saved to {final}")
            return True
        except Exception as e:
            print(f"  Error: {e}")
            return False

    def download_hesc_markers(self, genome='hg19'):
        bh = f"https://hgdownload.soe.ucsc.edu/goldenPath/{genome}/encodeDCC/wgEncodeBroadHistone"
        bt = f"https://hgdownload.soe.ucsc.edu/goldenPath/{genome}/encodeDCC/wgEncodeSydhTfbs"
        tracks = {
            'CTCF': f"{bh}/wgEncodeBroadHistoneH1hescCtcfStdPk.broadPeak.gz",
            'H3K4me1': f"{bh}/wgEncodeBroadHistoneH1hescH3k4me1StdPk.broadPeak.gz",
            'H3K4me3': f"{bh}/wgEncodeBroadHistoneH1hescH3k4me3StdPk.broadPeak.gz",
            'H3K27ac': f"{bh}/wgEncodeBroadHistoneH1hescH3k27acStdPk.broadPeak.gz",
        }
        rnapii_urls = [
            f"{bt}/wgEncodeSydhTfbsH1hescPol2StdPk.narrowPeak.gz",
            f"{bt}/wgEncodeSydhTfbsH1hescPol2IggmusPk.narrowPeak.gz",
            f"{bh}/wgEncodeBroadHistoneH1hescPol2bStdPk.broadPeak.gz",
        ]
        downloaded = {}
        for marker, url in tracks.items():
            local = f"{marker}_{genome}.bed"
            if (self.output_dir / local).exists():
                downloaded[marker] = str(self.output_dir / local)
                print(f"  {marker} already exists, skipping.")
            elif self.download_file(url, local):
                downloaded[marker] = str(self.output_dir / local)
        rnapii_local = f"RNAPII_{genome}.bed"
        if (self.output_dir / rnapii_local).exists():
            downloaded['RNAPII'] = str(self.output_dir / rnapii_local)
        else:
            for url in rnapii_urls:
                if self.download_file(url, rnapii_local):
                    downloaded['RNAPII'] = str(self.output_dir / rnapii_local)
                    break
        return downloaded


def prepare_chipseq_data(genome='hg19', output_dir='chipseq_data'):
    dl = ChIPSeqDownloader(output_dir)
    mf = dl.download_hesc_markers(genome)
    print(f"\nDownloaded {len(mf)} markers: {list(mf.keys())}")
    return mf


class ChIPSeqDataLoader:
    def __init__(self, genome='hg19'):
        self.genome = genome

    def load_custom_peaks(self, bed_file, chromosome=None):
        print(f"Loading peaks from {bed_file}...")
        with open(bed_file, 'r') as f:
            first = f.readline().strip()
            if not first:
                return pd.DataFrame(columns=['chr','start','end'])
            nc = len(first.split('\t'))
        names = ['chr','start','end','name','score','strand','signalValue','pValue','qValue','peak'][:nc]
        df = pd.read_csv(bed_file, sep='\t', header=None, names=names, comment='#')
        if chromosome and 'chr' in df.columns:
            df = df[df['chr'] == chromosome].copy()
        print(f"  Loaded {len(df)} peaks")
        return df


class BiologicalValidator:
    def __init__(self, resolution, chromosome, genome='hg19',
                 output_dir='biological_validation'):
        self.resolution = resolution
        self.chromosome = chromosome
        self.genome = genome
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chip_loader = ChIPSeqDataLoader(genome)
        self.markers: Dict[str, pd.DataFrame] = {}

    def load_marker_data(self, marker_files=None):
        if marker_files:
            for m, fp in marker_files.items():
                try:
                    df = self.chip_loader.load_custom_peaks(fp, self.chromosome)
                    if len(df) > 0: self.markers[m] = df
                except Exception as e:
                    print(f"  Warning: {m}: {e}")
        print(f"\nLoaded markers: {list(self.markers.keys())}")

    def peaks_to_bins(self, peaks_df, n_bins):
        counts = np.zeros(n_bins, dtype=int)
        for _, pk in peaks_df.iterrows():
            sb = max(0, min(int(pk['start']/self.resolution), n_bins-1))
            eb = max(0, min(int(pk['end']/self.resolution), n_bins-1))
            counts[sb:eb+1] += 1
        return counts

    def calculate_boundary_enrichment(self, tad_borders, marker_name, n_bins, window=10):
        if marker_name not in self.markers:
            raise ValueError(f"Marker {marker_name} not loaded")
        bp = self.peaks_to_bins(self.markers[marker_name], n_bins)
        boundaries = sorted({int(b) for tad in tad_borders for b in tad})
        profile = np.zeros(2*window+1)
        valid = 0
        for b in boundaries:
            if window <= b < n_bins - window:
                profile += bp[b-window:b+window+1]
                valid += 1
        if valid > 0: profile /= valid
        bpeaks = [bp[b] for b in boundaries if 0 <= b < n_bins]
        avg = float(np.mean(bpeaks)) if bpeaks else 0.0
        return {'enrichment_profile':profile, 'avg_peaks_per_bin':avg,
                'total_boundary_peaks':sum(bpeaks), 'n_boundaries':len(boundaries),
                'positions':np.arange(-window, window+1)}

    def validate_method(self, tad_borders, n_bins, method_name):
        rows = []
        for m in self.markers:
            e = self.calculate_boundary_enrichment(tad_borders, m, n_bins)
            rows.append({'Method':method_name,'Marker':m,
                         'Avg_Peaks_Per_Bin':e['avg_peaks_per_bin'],
                         'Total_Boundary_Peaks':e['total_boundary_peaks'],
                         'N_Boundaries':e['n_boundaries']})
        return pd.DataFrame(rows)

    def compare_methods(self, method_tads, n_bins):
        parts = []
        for name, borders in method_tads.items():
            print(f"\nValidating {name}...")
            parts.append(self.validate_method(borders, n_bins, name))
        combined = pd.concat(parts, ignore_index=True)
        out = self.output_dir / 'validation_results.csv'
        combined.to_csv(out, index=False)
        return combined

    def plot_enrichment_profiles(self, method_tads, n_bins, markers=None, window=500):
        markers = markers or list(self.markers.keys())
        nm = len(markers)
        if nm == 0: return
        fig, axes = plt.subplots(1, nm, figsize=(5*nm, 4))
        if nm == 1: axes = [axes]
        wb = int(window*1000/self.resolution)
        for idx, m in enumerate(markers):
            ax = axes[idx]
            for method, borders in method_tads.items():
                e = self.calculate_boundary_enrichment(borders, m, n_bins, window=wb)
                ax.plot(e['positions']*self.resolution/1000, e['enrichment_profile'],
                        label=method, linewidth=2)
            ax.axvline(0, color='black', ls='--', lw=1, alpha=0.5)
            ax.set_xlabel('Distance (kb)'); ax.set_ylabel(f'Avg {m} peaks/bin')
            ax.set_title(f'{m} Enrichment', fontweight='bold')
            ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enrichment_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_peaks_per_bin_comparison(self, method_tads, n_bins):
        markers = list(self.markers.keys())
        nm = len(markers)
        if nm == 0: return
        fig, axes = plt.subplots(1, nm, figsize=(4*nm, 4))
        if nm == 1: axes = [axes]
        for idx, m in enumerate(markers):
            ax = axes[idx]; methods, vals = [], []
            for method, borders in method_tads.items():
                e = self.calculate_boundary_enrichment(borders, m, n_bins, window=2)
                methods.append(method); vals.append(e['avg_peaks_per_bin'])
            colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
            bars = ax.bar(methods, vals, color=colors, edgecolor='black')
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2., b.get_height(),
                        f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=9)
            ax.set_ylabel(f'Avg {m} peaks/bin')
            ax.set_title(f'{m} at Boundaries', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'peaks_per_bin_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_validation_report(self, validation_results):
        lines = ["="*80, "BIOLOGICAL VALIDATION REPORT", "="*80, "",
                  f"Chromosome: {self.chromosome}",
                  f"Resolution: {self.resolution/1000:.0f} kb",
                  f"Genome: {self.genome}", "",
                  "ENRICHMENT SUMMARY", "-"*80]
        pivot = validation_results.pivot(index='Marker', columns='Method', values='Avg_Peaks_Per_Bin')
        lines.append(pivot.to_string()); lines.append("")
        lines += ["BEST METHOD PER MARKER", "-"*80]
        for m in validation_results['Marker'].unique():
            sub = validation_results[validation_results['Marker']==m]
            best = sub.loc[sub['Avg_Peaks_Per_Bin'].idxmax()]
            lines.append(f"  {m:15s}: {best['Method']:15s} ({best['Avg_Peaks_Per_Bin']:.3f})")
        lines += ["", "OVERALL RANKING", "-"*80]
        ranking = validation_results.groupby('Method')['Avg_Peaks_Per_Bin'].mean().sort_values(ascending=False)
        for r, (method, val) in enumerate(ranking.items(), 1):
            lines.append(f"  {r}. {method:15s}: {val:.3f}")
        lines += ["", "="*80]
        text = "\n".join(lines)
        (self.output_dir / 'validation_report.txt').write_text(text, encoding='utf-8')
        print(text); return text