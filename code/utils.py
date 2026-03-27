"""Utility functions."""

import numpy as np
from pathlib import Path


def convert_to_bed(input_file, output_file, first_col=2, second_col=4):
    try:
        data = np.loadtxt(input_file, skiprows=1)
        if data.ndim == 1: data = data.reshape(1,-1)
        starts = data[:, first_col-1].astype(int)
        stops = data[:, second_col-1].astype(int)
        with open(output_file, 'w') as f:
            for s, e in zip(starts, stops): f.write(f"{s}\t{e}\n")
        print(f"BED done: {output_file}")
    except Exception as e:
        print(f"Error: {e}")


def convert_tad_binid_to_bed(input_file, output_file, resolution, chromosome="chr1"):
    try:
        data = np.loadtxt(input_file, dtype=int)
        if data.ndim == 1: data = data.reshape(1,-1)
        with open(output_file, 'w') as f:
            for r in data:
                f.write(f"{chromosome}\t{r[0]*resolution}\t{r[1]*resolution}\n")
        print(f"BED done: {output_file}")
    except Exception as e:
        print(f"Error: {e}")


def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p