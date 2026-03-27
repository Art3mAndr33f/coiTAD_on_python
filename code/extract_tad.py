"""TAD extraction module."""

import numpy as np
from pathlib import Path
from typing import Tuple, List


class ExtractTAD:
    """Extract TADs from clustered data."""

    def __init__(self, chr_data, assign_cluster, radius,
                 resolution, algorithm, result_path):
        self.chr_data = chr_data
        self.assign_cluster = assign_cluster
        self.radius = radius
        self.resolution = resolution
        self.algorithm = algorithm
        self.result_path = Path(result_path)
        self.tad_path = self.result_path / 'TADs'
        self.tad_path.mkdir(exist_ok=True)

    def extract(self) -> List[float]:
        max_tad_size = 2000000
        try:
            new_borders, tad_sizes = self.find_tad(
                self.chr_data, self.assign_cluster, max_tad_size)
        except Exception as e:
            print(f"Error occurred: {e}")
            return [0, 0]
        if len(new_borders) > 0:
            avg_size = np.mean([b[1] - b[0] for b in new_borders])
            return [len(new_borders), avg_size]
        return [0, 0]

    def find_tad(self, chr_data, assign_cluster, max_tad_size):
        borders = []
        start = 0
        limit = len(assign_cluster)
        for i in range(1, limit):
            if assign_cluster[i] != assign_cluster[start]:
                borders.append([start, i - 1])
                start = i
        borders.append([start, limit - 1])

        min_num_tad = round(100000 / self.resolution)
        new_borders = []
        tad_sizes = []
        for border in borders:
            if border[1] <= chr_data.shape[0] and border[0] <= chr_data.shape[0]:
                if (border[1] - border[0] + 1) >= min_num_tad:
                    tad_size = (border[1] - border[0] + 1) * self.resolution
                    if tad_size > max_tad_size:
                        sub_tads = self.break_down_tad(
                            chr_data[border[0]:border[1]+1, border[0]:border[1]+1],
                            max_tad_size)
                        for st in sub_tads:
                            nt = [st[0] + border[0], st[1] + border[0]]
                            if nt[0] != nt[1]:
                                new_borders.append(nt)
                                tad_sizes.append((st[1]-st[0]+1)*self.resolution)
                    else:
                        if border[0] != border[1]:
                            new_borders.append(border)
                            tad_sizes.append(tad_size)

        zero_rows = self.find_zero_rows(chr_data)
        if len(new_borders) > 0:
            new_borders = [b for b in new_borders if b[1] not in zero_rows]
            self.save_tad_results(new_borders)
        return new_borders, tad_sizes

    def break_down_tad(self, tad_data, max_tad_size):
        num_bins = tad_data.shape[0]
        tad_size = num_bins * self.resolution
        if tad_size <= max_tad_size:
            return [[0, num_bins - 1]]
        max_bins = int(max_tad_size / self.resolution)
        new_tads = []
        start_bin = 0
        while start_bin < num_bins:
            end_bin = min(start_bin + max_bins - 1, num_bins - 1)
            if start_bin != end_bin:
                new_tads.append([start_bin, end_bin])
            start_bin = end_bin + 1
        return new_tads

    @staticmethod
    def find_zero_rows(chr_data):
        return [i for i in range(chr_data.shape[0]) if np.all(chr_data[i, :] == 0)]

    def save_tad_results(self, new_borders):
        fn = self.tad_path / f'{self.algorithm}_{self.radius}_TAD_BinID.txt'
        np.savetxt(fn, new_borders, fmt='%d')
        self.output_tad(new_borders)

    def output_tad(self, new_borders):
        fn = self.tad_path / f'{self.algorithm}_{self.radius}_domain.txt'
        with open(fn, 'w') as f:
            f.write(f"{'from.id':>6} {'from.cord':>12} {'to.id':>6} {'to.cord':>12}\n")
            for b in new_borders:
                f.write(f"{b[0]:>6} {b[0]*self.resolution:>12} "
                        f"{b[1]:>6} {b[1]*self.resolution:>12}\n")