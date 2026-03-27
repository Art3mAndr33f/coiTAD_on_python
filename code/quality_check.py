"""Quality check module for TAD assessment."""

import numpy as np
from pathlib import Path
from typing import List, Tuple


class QualityChecker:
    """Assess quality of identified TADs."""

    def __init__(self, chr_data, resolution, min_radius, max_radius,
                 tad_quality, result_path, quality_path, algorithm):
        self.chr_data = chr_data
        self.resolution = resolution
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.tad_quality = tad_quality
        self.result_path = Path(result_path)
        self.quality_path = Path(quality_path)
        self.algorithm = algorithm
        self.max_quality = 0
        self.best_radius = min_radius

    def check(self) -> int:
        tad_idx = 0
        for radius in range(self.min_radius, self.max_radius + 1):
            tad_file = self.result_path / 'TADs' / f'{self.algorithm}_{radius}_TAD_BinID.txt'
            if self.tad_quality[tad_idx][0] != 0:
                try:
                    tad_borders = np.loadtxt(tad_file, dtype=int)
                    if tad_borders.ndim == 1:
                        tad_borders = tad_borders.reshape(1, -1)
                except Exception:
                    print(f"Skipping radius {radius}")
                    tad_idx += 1
                    continue

                intra_scores, inter_scores = self.calculate_scores(tad_borders)
                if len(intra_scores) > 0:
                    avg_intra = np.mean(intra_scores)
                    avg_inter = np.mean(inter_scores)
                    avg_diff = avg_intra - avg_inter
                    print(f"Avg intra IF= {avg_intra:.6f}  inter IF= {avg_inter:.6f}  diff= {avg_diff:.6f}")
                    if avg_diff > self.max_quality:
                        self.max_quality = avg_diff
                        self.best_radius = radius
            else:
                print(f"Skipping radius {radius}")
            tad_idx += 1
        self.save_readme()
        return self.best_radius

    def calculate_scores(self, tad_borders):
        intra_scores, inter_scores = [], []
        n = len(tad_borders)
        for i in range(n):
            intra_scores.append(self.calc_intra(tad_borders[i]))
            if i == 0:
                d2 = tad_borders[i] if n == 1 else tad_borders[i + 1]
                inter, _ = self.calc_inter(tad_borders[i], d2)
            elif i < n - 1:
                s1, c1 = self.calc_inter(tad_borders[i-1], tad_borders[i])
                s2, c2 = self.calc_inter(tad_borders[i], tad_borders[i+1])
                inter = (s1 + s2) / (c1 + c2) if (c1 + c2) > 0 else 0
            else:
                inter, _ = self.calc_inter(tad_borders[i-1], tad_borders[i])
            inter_scores.append(inter)
        return intra_scores, inter_scores

    def calc_intra(self, domain):
        s, e = domain
        e = min(e, self.chr_data.shape[0] - 1)
        if s >= self.chr_data.shape[0]:
            return 0.0
        total, count = 0, 0
        for i in range(s, e + 1):
            for j in range(i + 1, e + 1):
                if i < self.chr_data.shape[0] and j < self.chr_data.shape[1]:
                    count += 1
                    total += self.chr_data[i, j]
        return total / count if count > 0 else 0.0

    def calc_inter(self, domain_i, domain_j):
        si, ei = domain_i
        sj, ej = domain_j
        total, count = 0, 0
        for i in range(si, sj):
            incr = i - si + 1
            c = 0
            for j in range(ei + 1, ej + 1):
                c += 1
                if i < self.chr_data.shape[0] and j < self.chr_data.shape[1]:
                    count += 1
                    total += self.chr_data[i, j]
                if c == incr:
                    break
        return total, count

    def save_readme(self):
        fn = self.quality_path / 'Readme.txt'
        with open(fn, 'w') as f:
            f.write(f'Recommended TAD = {self.algorithm}_{self.best_radius}_domain.txt '
                    f'with value {self.max_quality:.6f}\n')