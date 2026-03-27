"""Visualization: contact maps, TAD boundaries, statistics, mcool view."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Tuple, Optional


class CoiTADVisualizer:
    def __init__(self, contact_matrix_file, tad_file, resolution=40000,
                 output_dir="visualizations"):
        self.contact_matrix = np.loadtxt(contact_matrix_file)
        self.tad_borders = self._load(tad_file)
        self.resolution = resolution
        self.output_dir = Path(output_dir); self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def _load(f):
        try: data = np.loadtxt(f, dtype=int)
        except ValueError: data = np.loadtxt(f, skiprows=1, usecols=(0,2), dtype=int)
        if data.ndim == 1: data = data.reshape(1,-1)
        return [(int(r[0]),int(r[1])) for r in data]

    def _draw(self, ax, borders, color='blue', lw=2, alpha=0.8):
        for s,e in borders:
            ax.add_patch(patches.Rectangle((s-.5,s-.5),e-s+1,e-s+1,
                         lw=lw, edgecolor=color, facecolor='none', alpha=alpha))

    def _axes(self, ax, n):
        step = max(1, n//10)
        t = np.arange(0,n,step)
        lb = [f'{p*self.resolution/1e6:.1f}' for p in t]
        ax.set_xticks(t); ax.set_yticks(t)
        ax.set_xticklabels(lb); ax.set_yticklabels(lb)
        ax.set_xlabel('Position (Mb)'); ax.set_ylabel('Position (Mb)')

    def plot_contact_map_with_tads(self, save_name="contact_map_with_tads.png"):
        fig, ax = plt.subplots(figsize=(12,10))
        m = self.contact_matrix.copy(); m[m==0]=np.nan
        im = ax.imshow(m, cmap='Reds', norm=LogNorm(), interpolation='none', origin='upper')
        self._draw(ax, self.tad_borders)
        self._axes(ax, m.shape[0])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Hi-C with TADs ({len(self.tad_borders)} TADs)', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir/save_name, dpi=300, bbox_inches='tight'); plt.close()

    def plot_tad_statistics(self, save_name="tad_statistics.png"):
        sizes = [(e-s)*self.resolution/1000 for s,e in self.tad_borders]
        fig, axes = plt.subplots(1,3,figsize=(14,5))
        axes[0].hist(sizes,bins=20,color='steelblue',edgecolor='black',alpha=0.7)
        axes[0].axvline(np.median(sizes),color='red',ls='--',lw=2,label=f'Median: {np.median(sizes):.1f}kb')
        axes[0].set_xlabel('TAD Size (kb)'); axes[0].set_ylabel('Freq')
        axes[0].set_title('Size Distribution',fontweight='bold'); axes[0].legend()
        axes[1].boxplot(sizes,vert=True,patch_artist=True,boxprops=dict(facecolor='lightblue'),
                        medianprops=dict(color='red',lw=2))
        axes[1].set_ylabel('TAD Size (kb)'); axes[1].set_title('Size Stats',fontweight='bold')
        n = self.contact_matrix.shape[0]; cov = np.zeros(n)
        for s,e in self.tad_borders: cov[s:e+1]=1
        axes[2].fill_between(np.arange(n)*self.resolution/1e6,0,cov,alpha=0.5,color='green')
        axes[2].set_xlabel('Position (Mb)'); axes[2].set_ylabel('Coverage')
        axes[2].set_title('TAD Coverage',fontweight='bold'); axes[2].set_ylim(-0.1,1.1)
        plt.tight_layout()
        plt.savefig(self.output_dir/save_name, dpi=300, bbox_inches='tight'); plt.close()

    def generate_all_plots(self):
        print("  Contact map..."); self.plot_contact_map_with_tads()
        print("  Statistics..."); self.plot_tad_statistics()
        print(f"  Done: {self.output_dir}/")


def plot_mcool_with_tads(mcool_file, tad_file, chromosome="chr19",
                         resolution=50000, start=None, end=None, save=True):
    import cooler
    uri = f"{mcool_file}::resolutions/{resolution}"
    clr = cooler.Cooler(uri)
    if start and end:
        matrix = clr.matrix(balance=True).fetch(f"{chromosome}:{start}-{end}")
        start_bin = start // resolution
    else:
        matrix = clr.matrix(balance=True).fetch(chromosome)
        start_bin = 0
    matrix = np.nan_to_num(matrix, nan=0.0)
    tads = np.loadtxt(tad_file, dtype=int)
    if tads.ndim == 1: tads = tads.reshape(1,-1)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.matshow(matrix, cmap='RdYlBu_r', norm=plt.matplotlib.colors.LogNorm())
    for s,e in tads:
        rs,re = s-start_bin, e-start_bin
        if 0<=rs<matrix.shape[0] and 0<=re<matrix.shape[0]:
            ax.add_patch(patches.Rectangle((rs,rs),re-rs,re-rs,
                         fill=False,edgecolor='cyan',lw=2))
    ax.set_title(f'{chromosome} Hi-C with TADs', fontsize=14, pad=20)
    plt.tight_layout()
    if save:
        plt.savefig(f'{chromosome}_tads.png', dpi=300)
    plt.show()


def visualize_coitad_results(results_dir, data_dir, chromosome="chr19",
                             resolution=40000, best_radius=2, algorithm="HDBSCAN"):
    cm = f"{data_dir}/{chromosome}_{resolution//1000}kb.hic"
    tf = f"{results_dir}/TADs/{algorithm}_{best_radius}_TAD_BinID.txt"
    viz = CoiTADVisualizer(cm, tf, resolution, f"{results_dir}/visualizations")
    viz.generate_all_plots()
    return viz