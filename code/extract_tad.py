"""TAD extraction module"""

import numpy as np
from pathlib import Path
from typing import Tuple, List


class ExtractTAD:
    """Extract TADs from clustered data"""
    
    def __init__(self,
                 chr_data: np.ndarray,
                 assign_cluster: np.ndarray,
                 radius: int,
                 resolution: int,
                 algorithm: str,
                 result_path: Path):
        """
        Initialize TAD extractor
        
        Args:
            chr_data: Hi-C contact matrix
            assign_cluster: Cluster assignments
            radius: Current radius
            resolution: Data resolution
            algorithm: Clustering algorithm name
            result_path: Path to save results
        """
        self.chr_data = chr_data
        self.assign_cluster = assign_cluster
        self.radius = radius
        self.resolution = resolution
        self.algorithm = algorithm
        self.result_path = Path(result_path)
        
        # Create TADs directory
        self.tad_path = self.result_path / 'TADs'
        self.tad_path.mkdir(exist_ok=True)
        
    def extract(self) -> List[float]:
        """
        Extract TADs from clusters
        
        Returns:
            Quality metrics [num_tads, avg_size]
        """
        max_tad_size = 2000000
        
        try:
            new_borders, tad_sizes = self.find_tad(
                self.chr_data,
                self.assign_cluster,
                max_tad_size
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            return [0, 0]
        
        if len(new_borders) > 0:
            avg_size = np.mean([b[1] - b[0] for b in new_borders])
            quality = [len(new_borders), avg_size]
        else:
            quality = [0, 0]
            
        return quality
        
    def find_tad(self,
                 chr_data: np.ndarray,
                 assign_cluster: np.ndarray,
                 max_tad_size: int) -> Tuple[List, List]:
        """
        Find TAD boundaries from cluster assignments
        
        Args:
            chr_data: Contact matrix
            assign_cluster: Cluster assignments
            max_tad_size: Maximum TAD size
            
        Returns:
            Tuple of (borders, tad_sizes)
        """
        # Find borders
        borders = []
        start = 0
        limit = len(assign_cluster)
        
        for i in range(1, limit):
            if assign_cluster[i] != assign_cluster[start]:
                borders.append([start, i - 1])
                start = i
        borders.append([start, limit - 1])
        
        # Filter by minimum size (100kb)
        min_num_tad = round(100000 / self.resolution)
        new_borders = []
        tad_sizes = []
        
        for border in borders:
            if border[1] <= chr_data.shape[0] and border[0] <= chr_data.shape[0]:
                if (border[1] - border[0] + 1) >= min_num_tad:
                    tad_size = (border[1] - border[0] + 1) * self.resolution
                    
                    if tad_size > max_tad_size:
                        # Break down large TAD
                        sub_tads = self.break_down_tad(
                            chr_data[border[0]:border[1]+1, border[0]:border[1]+1],
                            max_tad_size
                        )
                        for sub_tad in sub_tads:
                            new_tad = [sub_tad[0] + border[0], sub_tad[1] + border[0]]
                            if new_tad[0] != new_tad[1]:
                                new_borders.append(new_tad)
                                tad_sizes.append((sub_tad[1] - sub_tad[0] + 1) * self.resolution)
                    else:
                        if border[0] != border[1]:
                            new_borders.append(border)
                            tad_sizes.append(tad_size)
        
        # Remove zero rows
        zero_rows = self.find_zero_rows(chr_data)
        
        # Filter borders
        if len(new_borders) > 0:
            new_borders_domain = []
            for border in new_borders:
                if border[1] not in zero_rows:
                    new_borders_domain.append(border)
            
            new_borders = new_borders_domain
            
            # Save to file
            self.save_tad_results(new_borders)
            
        return new_borders, tad_sizes
        
    def break_down_tad(self, tad_data: np.ndarray, max_tad_size: int) -> List:
        """
        Break down large TAD into smaller ones
        
        Args:
            tad_data: TAD contact matrix
            max_tad_size: Maximum TAD size
            
        Returns:
            List of sub-TAD borders
        """
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
    def find_zero_rows(chr_data: np.ndarray) -> List[int]:
        """Find rows that are all zeros"""
        zero_rows = []
        for i in range(chr_data.shape[0]):
            if np.all(chr_data[i, :] == 0):
                zero_rows.append(i)
        return zero_rows
        
    def save_tad_results(self, new_borders: List):
        """Save TAD results to file"""
        # Save TAD bin IDs
        filename = self.tad_path / f'{self.algorithm}_{self.radius}_TAD_BinID.txt'
        np.savetxt(filename, new_borders, fmt='%d')
        
        # Save domain file
        self.output_tad(new_borders)
        
    def output_tad(self, new_borders: List):
        """
        Output TAD coordinates to file
        
        Args:
            new_borders: List of TAD borders
        """
        filename = self.tad_path / f'{self.algorithm}_{self.radius}_domain.txt'
        
        with open(filename, 'w') as f:
            f.write(f"{'from.id':>6} {'from.cord':>12} {'to.id':>6} {'to.cord':>12}\n")
            for border in new_borders:
                from_id = border[0]
                from_cord = from_id * self.resolution
                to_id = border[1]
                to_cord = to_id * self.resolution
                f.write(f"{from_id:>6} {from_cord:>12} {to_id:>6} {to_cord:>12}\n")