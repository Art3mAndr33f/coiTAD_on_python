# interactive_visualization.py
"""Интерактивная визуализация с использованием cooler/cooltools"""

import cooler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_mcool_with_tads(
    mcool_file: str,
    tad_file: str,
    chromosome: str = "chr19",
    resolution: int = 50000,
    start: int = None,
    end: int = None
):
    """
    Визуализация напрямую из .mcool с TAD
    """
    # Загружаем данные
    uri = f"{mcool_file}::resolutions/{resolution}"
    clr = cooler.Cooler(uri)
    
    # Извлекаем регион
    if start and end:
        region = f"{chromosome}:{start}-{end}"
        matrix = clr.matrix(balance=True).fetch(region)
        start_bin = start // resolution
    else:
        matrix = clr.matrix(balance=True).fetch(chromosome)
        start_bin = 0
    
    matrix = np.nan_to_num(matrix, nan=0.0)
    
    # Загружаем TAD
    tad_borders = np.loadtxt(tad_file, dtype=int)
    if tad_borders.ndim == 1:
        tad_borders = tad_borders.reshape(1, -1)
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.matshow(matrix, cmap='RdYlBu_r', 
                    norm=plt.matplotlib.colors.LogNorm())
    
    # Рисуем TAD
    for start_tad, end_tad in tad_borders:
        # Корректируем относительно окна
        rel_start = start_tad - start_bin
        rel_end = end_tad - start_bin
        
        if 0 <= rel_start < matrix.shape[0] and 0 <= rel_end < matrix.shape[0]:
            rect = Rectangle(
                (rel_start, rel_start),
                rel_end - rel_start,
                rel_end - rel_start,
                fill=False,
                edgecolor='cyan',
                linewidth=2
            )
            ax.add_patch(rect)
    
    plt.colorbar(im, ax=ax, label='Contact Frequency')
    ax.set_title(f'{chromosome} Hi-C with TADs', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{chromosome}_tads_interactive.png', dpi=300)
    plt.show()


# Использование
if __name__ == "__main__":
    plot_mcool_with_tads(
        mcool_file="4DNFI52OLNJ4.mcool",
        tad_file="coitad_output/results/TADs/HDBSCAN_2_TAD_BinID.txt",
        chromosome="chr19",
        resolution=50000
    )
