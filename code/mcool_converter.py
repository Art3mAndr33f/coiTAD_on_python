"""Converter for .mcool files to text matrices."""

import cooler
import numpy as np
from pathlib import Path


class McoolConverter:
    def __init__(self, mcool_file: str):
        self.mcool_file = mcool_file

    def list_resolutions(self):
        try:
            resolutions = cooler.fileops.list_coolers(self.mcool_file)
            print("Available resolutions:")
            for r in resolutions:
                print(f"  - {r}")
            return resolutions
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

    def extract_chromosome(self, chromosome, resolution,
                           output_file=None, normalize=True, balance=True):
        uri = f"{self.mcool_file}::resolutions/{resolution}"
        try:
            clr = cooler.Cooler(uri)
            available = clr.chromnames
            if chromosome not in available:
                if f'chr{chromosome}' in available:
                    chromosome = f'chr{chromosome}'
                elif chromosome.replace('chr', '') in available:
                    chromosome = chromosome.replace('chr', '')
                else:
                    raise ValueError(f"Chromosome {chromosome} not found. "
                                     f"Available: {available}")

            print(f"Extracting {chromosome} at {resolution}bp...")
            if balance and 'weight' in clr.bins().columns:
                matrix = clr.matrix(balance=True).fetch(chromosome)
            else:
                matrix = clr.matrix(balance=False).fetch(chromosome)

            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            if normalize and not balance:
                mx = np.max(matrix)
                if mx > 0:
                    matrix = matrix / mx

            print(f"Matrix size: {matrix.shape}")
            print(f"Non-zero: {np.count_nonzero(matrix)} "
                  f"({100*np.count_nonzero(matrix)/matrix.size:.2f}%)")

            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(output_file, matrix, fmt='%.6f')
                print(f"Saved to {output_file}")
            return matrix
        except Exception as e:
            print(f"Error: {e}")
            raise

    def get_chromosome_info(self, resolution):
        uri = f"{self.mcool_file}::resolutions/{resolution}"
        try:
            clr = cooler.Cooler(uri)
            chromsizes = clr.chromsizes
            print(f"\n{'Chromosome':<15} {'Size (bp)':<15} {'Bins':<10}")
            print("-" * 40)
            for chrom, size in chromsizes.items():
                print(f"{chrom:<15} {size:<15,} {size // resolution:<10,}")
            return chromsizes
        except Exception as e:
            print(f"Error: {e}")
            return None