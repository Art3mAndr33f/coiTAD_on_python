# coitad_main.py
"""
Implementation of coiTAD algorithm
Translated from MATLAB to Python
@author: Translation by Assistant
Original authors: Oluwatosin Oluwadare/Drew Houchens
UCCS Bioinformatics Lab
"""

import numpy as np
import os
import warnings
from pathlib import Path
import hdbscan
from typing import Tuple, List

warnings.filterwarnings('ignore')


class CoiTAD:
    """Main class for coiTAD algorithm"""
    
    def __init__(self, 
                 filepath: str,
                 feature_filepath: str,
                 filename: str,
                 chromo: str = 'chr',
                 resolution: int = 40000,
                 max_tad_size: int = 800000,
                 output_folder: str = 'data_Results'):
        """
        Initialize coiTAD parameters
        
        Args:
            filepath: Path to input data file
            feature_filepath: Path to store generated features
            filename: Name of input file
            chromo: Chromosome name
            resolution: Data resolution in base pairs (e.g., 40000 for 40kb)
            max_tad_size: Maximum TAD size in base pairs
            output_folder: Name of output folder
        """
        self.filepath = Path(filepath)
        self.feature_filepath = Path(feature_filepath)
        self.filename = filename
        self.chromo = chromo
        self.resolution = resolution
        self.max_tad_size = max_tad_size
        self.output_folder = output_folder
        self.algorithm = 'HDBSCAN'
        
        # Derived parameters
        self.kb = 1000
        self.min_radius = 2
        self.max_radius = int((max_tad_size / resolution) + 10)
        self.max_quality = 0
        self.best_radius = None
        
        # Load data
        self.chr_data = None
        self.result_path = None
        self.out_path = None
        
    def load_data(self):
        """Load Hi-C contact matrix"""
        full_path = self.filepath / self.filename
        print(f"Loading data from {full_path}...")
        self.chr_data = np.loadtxt(full_path)
        print("Data set loaded.")
        
        # Create output directory
        self.result_path = Path(self.output_folder)
        self.result_path.mkdir(exist_ok=True)
        
    def run(self):
        """Main execution pipeline"""
        # Load data
        self.load_data()
        
        # Generate features
        print("Generating features...")
        self.generate_features()
        
        # Perform clustering
        print("Performing clustering...")
        radius_optimal_clusters = self.perform_clustering()
        
        # Process clusters and extract TADs
        print("=" * 60)
        print("Quality Assessment")
        print("=" * 60)
        tad_quality = self.process_clusters(radius_optimal_clusters)
        
        # Quality check
        self.quality_check(tad_quality)
        
        print("=" * 60)
        print("Quality Assessment Completed")
        print("=" * 60)
        print(f"Find the results in the Quality/ directory")
        print(f"Recommended radius = {self.best_radius}")
        print(f"Find the TADs identified in the TAD/ directory")
        print("=" * 60)
        print("coiTAD Completed")
        print("=" * 60)
        
    def generate_features(self):
        """Generate circle of influence features for all radii"""
        from feature_generation import FeatureGenerator
        
        generator = FeatureGenerator(
            contact_matrix=self.chr_data,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            output_folder=self.feature_filepath
        )
        generator.generate_all_features()
        
    def perform_clustering(self) -> np.ndarray:
        """
        Perform HDBSCAN clustering for each radius feature
        
        Returns:
            Matrix of cluster labels for each radius
        """
        # Determine maximum length
        max_length = 0
        for radius in range(self.min_radius, self.max_radius + 1):
            file_path = self.feature_filepath / f'feature_radius_{radius}.txt'
            radius_data = np.loadtxt(file_path)
            
            # Run HDBSCAN
            clusterer = hdbscan.HDBSCAN(metric='euclidean')
            clusterer.fit(radius_data)
            
            if len(clusterer.labels_) > max_length:
                max_length = len(clusterer.labels_)
        
        # Initialize matrix
        num_radii = self.max_radius - self.min_radius + 1
        radius_optimal_clusters = np.zeros((max_length, num_radii))
        
        # Second pass: store cluster labels
        for radius in range(self.min_radius, self.max_radius + 1):
            print(f"Processing radius = {radius}")
            file_path = self.feature_filepath / f'feature_radius_{radius}.txt'
            radius_data = np.loadtxt(file_path)
            
            # Run HDBSCAN
            clusterer = hdbscan.HDBSCAN(metric='euclidean')
            clusterer.fit(radius_data)
            clusters = clusterer.labels_
            
            # Store in matrix
            col_idx = radius - self.min_radius
            radius_optimal_clusters[:len(clusters), col_idx] = clusters
            
        return radius_optimal_clusters
        
    def process_clusters(self, radius_optimal_clusters: np.ndarray) -> List:
        """
        Process clusters to extract TADs
        
        Args:
            radius_optimal_clusters: Matrix of cluster labels
            
        Returns:
            List of TAD quality metrics
        """
        from extract_tad import ExtractTAD
        
        tad_quality = []
        
        for radius_idx in range(radius_optimal_clusters.shape[1]):
            radius = self.min_radius + radius_idx
            print(f"Result when radius = {radius}")
            
            # Order TAD numbers
            clusters = radius_optimal_clusters[:, radius_idx]
            assign_cluster = self.order_tad_num(clusters)
            
            # Extract TAD
            extractor = ExtractTAD(
                chr_data=self.chr_data,
                assign_cluster=assign_cluster,
                radius=radius,
                resolution=self.resolution,
                algorithm=self.algorithm,
                result_path=self.result_path
            )
            
            quality = extractor.extract()
            tad_quality.append(quality)
            
        return tad_quality
        
    @staticmethod
    def order_tad_num(found_tad: np.ndarray) -> np.ndarray:
        """
        Number the TAD clusters found
        
        Args:
            found_tad: Array of cluster labels
            
        Returns:
            Numbered TAD array
        """
        length = len(found_tad)
        assign = np.zeros(length, dtype=int)
        count = 1
        assign[0] = count
        
        for i in range(1, length):
            if found_tad[i-1] != found_tad[i]:
                count += 1
            assign[i] = count
            
        return assign
        
    def quality_check(self, tad_quality: List):
        """
        Perform quality assessment on identified TADs
        
        Args:
            tad_quality: List of quality metrics for each radius
        """
        from quality_check import QualityChecker
        
        # Create Quality output directory
        quality_path = self.result_path / 'Quality'
        quality_path.mkdir(exist_ok=True)
        
        checker = QualityChecker(
            chr_data=self.chr_data,
            resolution=self.resolution,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            tad_quality=tad_quality,
            result_path=self.result_path,
            quality_path=quality_path,
            algorithm=self.algorithm
        )
        
        self.best_radius = checker.check()
        self.max_quality = checker.max_quality