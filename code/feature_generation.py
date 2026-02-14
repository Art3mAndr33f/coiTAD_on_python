"""Feature generation module for coiTAD"""

import numpy as np
from pathlib import Path
from typing import Tuple


class FeatureGenerator:
    """Generates circle of influence features from Hi-C contact matrix"""
    
    def __init__(self, 
                 contact_matrix: np.ndarray,
                 min_radius: int,
                 max_radius: int,
                 output_folder: Path):
        """
        Initialize feature generator
        
        Args:
            contact_matrix: Hi-C contact matrix
            min_radius: Minimum radius for circle of influence
            max_radius: Maximum radius for circle of influence
            output_folder: Path to save generated features
        """
        self.contact_matrix = contact_matrix
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.output_folder = Path(output_folder)
        
        # Create output folder
        self.output_folder.mkdir(exist_ok=True)
        
    def generate_all_features(self):
        """Generate features for all radii"""
        for radius in range(self.min_radius, self.max_radius + 1):
            print(f"Generating features for radius = {radius}")
            clustering_input = self.create_entire_feature(radius)
            
            # Save to file
            filename = f'feature_radius_{radius}.txt'
            filepath = self.output_folder / filename
            np.savetxt(filepath, clustering_input)
            
    def create_entire_feature(self, radius: int) -> np.ndarray:
        """
        Create feature matrix for entire diagonal
        
        Args:
            radius: Current radius
            
        Returns:
            Feature matrix
        """
        num_rows, num_cols = self.contact_matrix.shape
        diagonal_len = min(num_rows, num_cols)
        
        clustering_input = []
        
        for index in range(diagonal_len):
            starting_point = (index, index)
            feature_vector = self.fill_final_matrix(starting_point, radius)
            clustering_input.append(feature_vector)
            
        return np.array(clustering_input)
        
    def fill_final_matrix(self, starting_point: Tuple[int, int], radius: int) -> np.ndarray:
        """
        Fill feature vector for a single diagonal point
        
        Args:
            starting_point: (row, col) position on diagonal
            radius: Circle of influence radius
            
        Returns:
            Feature vector for this point
        """
        num_rows, num_cols = self.contact_matrix.shape
        row, col = starting_point
        
        # Initialize directions
        c = self._get_safe_value(row, col)
        tC = [self._get_safe_value(row - 1, col)]
        tR = [self._get_safe_value(row - 1, col + 1)]
        r = [self._get_safe_value(row, col + 1)]
        bR = [self._get_safe_value(row + 1, col + 1)]
        b = [self._get_safe_value(row + 1, col)]
        bL = [self._get_safe_value(row + 1, col - 1)]
        l = [self._get_safe_value(row, col - 1)]
        tL = [self._get_safe_value(row - 1, col - 1)]
        
        # Expand for radius > 2
        for operator in range(2, radius + 1):
            tC.append(self._get_safe_value(row - operator, col))
            tR.append(self._get_safe_value(row - operator, col + operator))
            r.append(self._get_safe_value(row, col + operator))
            bR.append(self._get_safe_value(row + operator, col + operator))
            b.append(self._get_safe_value(row + operator, col))
            bL.append(self._get_safe_value(row + operator, col - operator))
            l.append(self._get_safe_value(row, col - operator))
            tL.append(self._get_safe_value(row - operator, col - operator))
        
        # Full circle feature
        feature_vector = tL + tC + tR + r + [c] + l + bL + b + bR
        
        # For semi-circle feature, use:
        # feature_vector = tL + tC + tR + r + [c] + bR
        
        return np.array(feature_vector)
        
    def _get_safe_value(self, row: int, col: int) -> float:
        """
        Get value from matrix with bounds checking
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Matrix value or 0 if out of bounds
        """
        num_rows, num_cols = self.contact_matrix.shape
        
        if row < 0 or row >= num_rows or col < 0 or col >= num_cols:
            return 0.0
        return self.contact_matrix[row, col]