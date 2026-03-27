"""Feature generation module for coiTAD."""

import numpy as np
from pathlib import Path
from typing import Tuple


class FeatureGenerator:
    """Generates circle of influence features from Hi-C contact matrix."""

    def __init__(self, contact_matrix: np.ndarray,
                 min_radius: int, max_radius: int,
                 output_folder):
        self.contact_matrix = contact_matrix
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

    def generate_all_features(self):
        for radius in range(self.min_radius, self.max_radius + 1):
            print(f"Generating features for radius = {radius}")
            clustering_input = self.create_entire_feature(radius)
            filepath = self.output_folder / f'feature_radius_{radius}.txt'
            np.savetxt(filepath, clustering_input)

    def create_entire_feature(self, radius: int) -> np.ndarray:
        num_rows, num_cols = self.contact_matrix.shape
        diagonal_len = min(num_rows, num_cols)
        clustering_input = []
        for index in range(diagonal_len):
            feature_vector = self.fill_final_matrix((index, index), radius)
            clustering_input.append(feature_vector)
        return np.array(clustering_input)

    def fill_final_matrix(self, starting_point: Tuple[int, int],
                          radius: int) -> np.ndarray:
        row, col = starting_point
        c = self._get_safe_value(row, col)
        tC = [self._get_safe_value(row - 1, col)]
        tR = [self._get_safe_value(row - 1, col + 1)]
        r = [self._get_safe_value(row, col + 1)]
        bR = [self._get_safe_value(row + 1, col + 1)]
        b = [self._get_safe_value(row + 1, col)]
        bL = [self._get_safe_value(row + 1, col - 1)]
        l = [self._get_safe_value(row, col - 1)]
        tL = [self._get_safe_value(row - 1, col - 1)]

        for op in range(2, radius + 1):
            tC.append(self._get_safe_value(row - op, col))
            tR.append(self._get_safe_value(row - op, col + op))
            r.append(self._get_safe_value(row, col + op))
            bR.append(self._get_safe_value(row + op, col + op))
            b.append(self._get_safe_value(row + op, col))
            bL.append(self._get_safe_value(row + op, col - op))
            l.append(self._get_safe_value(row, col - op))
            tL.append(self._get_safe_value(row - op, col - op))

        feature_vector = tL + tC + tR + r + [c] + l + bL + b + bR
        return np.array(feature_vector)

    def _get_safe_value(self, row: int, col: int) -> float:
        nr, nc = self.contact_matrix.shape
        if row < 0 or row >= nr or col < 0 or col >= nc:
            return 0.0
        return self.contact_matrix[row, col]