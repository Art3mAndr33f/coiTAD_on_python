"""Quality check module for TAD assessment"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
from scipy.stats import pearsonr


class QualityChecker:
    """Assess quality of identified TADs"""
    
    def __init__(self,
                 chr_data: np.ndarray,
                 resolution: int,
                 min_radius: int,
                 max_radius: int,
                 tad_quality: List,
                 result_path: Path,
                 quality_path: Path,
                 algorithm: str):
        """
        Initialize quality checker
        
        Args:
            chr_data: Hi-C contact matrix
            resolution: Data resolution
            min_radius: Minimum radius
            max_radius: Maximum radius
            tad_quality: List of TAD quality metrics
            result_path: Path to results
            quality_path: Path to quality results
            algorithm: Algorithm name
        """
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
        """
        Perform quality check
        
        Returns:
            Best radius
        """
        mean_intra = []
        mean_inter = []
        mean_diff = []
        
        tad_idx = 0
        
        for radius in range(self.min_radius, self.max_radius + 1):
            tad_file = self.result_path / 'TADs' / f'{self.algorithm}_{radius}_TAD_BinID.txt'
            
            if self.tad_quality[tad_idx][0] != 0:
                # Load TAD borders
                try:
                    tad_borders = np.loadtxt(tad_file, dtype=int)
                    if tad_borders.ndim == 1:
                        tad_borders = tad_borders.reshape(1, -1)
                except:
                    print(f"Skipping radius {radius}")
                    tad_idx += 1
                    continue
                
                # Calculate intra and inter scores
                intra_scores, inter_scores = self.calculate_scores(tad_borders)
                
                if len(intra_scores) > 0:
                    avg_intra = np.mean(intra_scores)
                    avg_inter = np.mean(inter_scores)
                    avg_diff = avg_intra - avg_inter
                    
                    print(f"The average IF for over each TD= {avg_intra:.6f}")
                    print(f"The average_Inter IF for over all TDs= {avg_inter:.6f}")
                    print(f"The average difference= {avg_diff:.6f}")
                    
                    mean_intra.append(avg_intra)
                    mean_inter.append(avg_inter)
                    mean_diff.append(avg_diff)
                    
                    if avg_diff > self.max_quality:
                        self.max_quality = avg_diff
                        self.best_radius = radius
            else:
                print(f"Skipping radius {radius}")
                
            tad_idx += 1
        
        # Save readme
        self.save_readme()
        
        return self.best_radius
        
    def calculate_scores(self, tad_borders: np.ndarray) -> Tuple[List, List]:
        """
        Calculate intra and inter TAD scores
        
        Args:
            tad_borders: Array of TAD borders
            
        Returns:
            Tuple of (intra_scores, inter_scores)
        """
        intra_scores = []
        inter_scores = []
        
        num_tads = len(tad_borders)
        
        for i in range(num_tads):
            # Calculate intra-TAD score
            intra = self.calc_intra(tad_borders[i])
            intra_scores.append(intra)
            
            # Calculate inter-TAD score
            if i == 0:
                if num_tads == 1:
                    domain2 = tad_borders[i]
                else:
                    domain2 = tad_borders[i + 1]
                inter, _ = self.calc_inter(tad_borders[i], domain2)
                
            elif i > 0 and i < num_tads - 1:
                domain_prev = tad_borders[i - 1]
                domain_next = tad_borders[i + 1]
                sum1, count1 = self.calc_inter(domain_prev, tad_borders[i])
                sum2, count2 = self.calc_inter(tad_borders[i], domain_next)
                if (count1 + count2) > 0:
                    inter = (sum1 + sum2) / (count1 + count2)
                else:
                    inter = 0
                    
            elif i == num_tads - 1:
                domain_prev = tad_borders[i - 1]
                inter, _ = self.calc_inter(domain_prev, tad_borders[i])
                
            inter_scores.append(inter)
            
        return intra_scores, inter_scores
        
    def calc_intra(self, domain: np.ndarray) -> float:
        """
        Calculate intra-TAD contact frequency
        
        Args:
            domain: TAD border [start, end]
            
        Returns:
            Average intra-TAD contact frequency
        """
        start, end = domain
        
        if end >= self.chr_data.shape[0]:
            end = self.chr_data.shape[0] - 1
        if start >= self.chr_data.shape[0]:
            return 0.0
            
        total_sum = 0
        count = 0
        
        for i in range(start, end + 1):
            for j in range(i + 1, end + 1):
                if i < self.chr_data.shape[0] and j < self.chr_data.shape[1]:
                    count += 1
                    total_sum += self.chr_data[i, j]
        
        if total_sum > 0 and count > 0:
            return total_sum / count
        else:
            return 0.0
            
    def calc_inter(self, domain_i: np.ndarray, domain_j: np.ndarray) -> Tuple[float, int]:
        """
        Calculate inter-TAD contact frequency
        
        Args:
            domain_i: First TAD border
            domain_j: Second TAD border
            
        Returns:
            Tuple of (sum, count)
        """
        start_i, end_i = domain_i
        start_j, end_j = domain_j
        
        total_sum = 0
        count = 0
        
        for i in range(start_i, start_j):
            incr = i - start_i + 1
            c = 0
            for j in range(end_i + 1, end_j + 1):
                c += 1
                if i < self.chr_data.shape[0] and j < self.chr_data.shape[1]:
                    count += 1
                    total_sum += self.chr_data[i, j]
                if c == incr:
                    break
        
        return total_sum, count
        
    def save_readme(self):
        """Save readme with best result"""
        readme_file = self.quality_path / f'Readme.txt'
        with open(readme_file, 'w') as f:
            f.write(f'Recommended TAD = {self.algorithm}_{self.best_radius}_domain.txt ')
            f.write(f'with value {self.max_quality:.6f}\n')