"""
Utility functions for coiTAD project.
"""

import numpy as np
from pathlib import Path
from typing import Optional


def convert_to_bed(input_file: str,
                   output_file: str,
                   first_col: int = 2,
                   second_col: int = 4):
    """
    Convert TAD domain file to BED format.

    Args:
        input_file:  path to TAD domain file (with header)
        output_file: path to output .bed
        first_col:   1-indexed column for start position
        second_col:  1-indexed column for end position
    """
    try:
        data = np.loadtxt(input_file, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        starts = data[:, first_col - 1].astype(int)
        stops = data[:, second_col - 1].astype(int)

        with open(output_file, 'w') as f:
            for s, e in zip(starts, stops):
                f.write(f"{s}\t{e}\n")

        print(f"BED conversion done: {output_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")


def convert_tad_binid_to_bed(input_file: str,
                              output_file: str,
                              resolution: int,
                              chromosome: str = "chr1"):
    """
    Convert TAD_BinID.txt → standard 3-column BED (chr, start_bp, end_bp).

    Args:
        input_file:  TAD_BinID.txt (two int columns: start_bin  end_bin)
        output_file: output .bed path
        resolution:  Hi-C resolution in bp
        chromosome:  chromosome name to put in first column
    """
    try:
        data = np.loadtxt(input_file, dtype=int)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        with open(output_file, 'w') as f:
            for row in data:
                start_bp = row[0] * resolution
                end_bp = row[1] * resolution
                f.write(f"{chromosome}\t{start_bp}\t{end_bp}\n")

        print(f"BED conversion done: {output_file}")
    except Exception as e:
        print(f"Error: {e}")


def find_zero_rows(matrix: np.ndarray):
    """Return list of row indices that are entirely zero."""
    return [i for i in range(matrix.shape[0]) if np.all(matrix[i] == 0)]


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p