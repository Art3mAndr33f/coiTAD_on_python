# coitad_optics.py
"""
coiTAD with OPTICS clustering algorithm
Модернизированная версия с использованием OPTICS вместо HDBSCAN
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import OPTICS
import warnings

warnings.filterwarnings('ignore')


class CoiTAD_OPTICS:
    """coiTAD with OPTICS clustering"""
    
    def __init__(self,
                 filepath: str,
                 feature_filepath: str,
                 filename: str,
                 chromo: str = 'chr',
                 resolution: int = 50000,
                 max_tad_size: int = 800000,
                 output_folder: str = 'data_Results_OPTICS',
                 min_samples: int = 5,
                 xi: float = 0.05,
                 min_cluster_size: float = 0.05):
        """
        Args:
            min_samples: Минимальное количество образцов в окрестности
            xi: Определяет минимальную крутизну кривой достижимости
            min_cluster_size: Минимальный размер кластера (доля от общего числа точек)
        """
        self.filepath = Path(filepath)
        self.feature_filepath = Path(feature_filepath)
        self.filename = filename
        self.chromo = chromo
        self.resolution = resolution
        self.max_tad_size = max_tad_size
        self.output_folder = output_folder
        self.algorithm = 'OPTICS'
        
        # OPTICS parameters
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        
        # Derived parameters
        self.kb = 1000
        self.min_radius = 2
        self.max_radius = int((max_tad_size / resolution) + 10)
        self.max_quality = 0
        self.best_radius = None
        
        # Load data
        self.chr_data = None
        self.result_path = None
        
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
        from feature_generation import FeatureGenerator
        from extract_tad import ExtractTAD
        from quality_check import QualityChecker
        
        # Load data
        self.load_data()
        
        # Generate features
        print("Generating features...")
        generator = FeatureGenerator(
            contact_matrix=self.chr_data,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            output_folder=self.feature_filepath
        )
        generator.generate_all_features()
        
        # Perform clustering with OPTICS
        print("Performing OPTICS clustering...")
        radius_optimal_clusters = self.perform_optics_clustering()
        
        # Process clusters and extract TADs
        print("=" * 60)
        print("Quality Assessment")
        print("=" * 60)
        tad_quality = self.process_clusters(radius_optimal_clusters)
        
        # Quality check
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
        
        print("=" * 60)
        print("Quality Assessment Completed")
        print("=" * 60)
        print(f"Recommended radius = {self.best_radius}")
        print(f"Find the TADs identified in the TAD/ directory")
        print("=" * 60)
        print("coiTAD-OPTICS Completed")
        print("=" * 60)
        
    def perform_optics_clustering(self) -> np.ndarray:
        """
        Perform OPTICS clustering for each radius feature
        
        Returns:
            Matrix of cluster labels for each radius
        """
        # Determine maximum length
        max_length = 0
        for radius in range(self.min_radius, self.max_radius + 1):
            file_path = self.feature_filepath / f'feature_radius_{radius}.txt'
            radius_data = np.loadtxt(file_path)
            
            if len(radius_data.shape) == 1:
                radius_data = radius_data.reshape(-1, 1)
            
            if radius_data.shape[0] > max_length:
                max_length = radius_data.shape[0]
        
        # Initialize matrix
        num_radii = self.max_radius - self.min_radius + 1
        radius_optimal_clusters = np.zeros((max_length, num_radii))
        
        # Clustering for each radius
        for radius in range(self.min_radius, self.max_radius + 1):
            print(f"Processing radius = {radius}")
            file_path = self.feature_filepath / f'feature_radius_{radius}.txt'
            radius_data = np.loadtxt(file_path)
            
            if len(radius_data.shape) == 1:
                radius_data = radius_data.reshape(-1, 1)
            
            # Calculate min_cluster_size based on data size
            min_cluster_size_abs = max(2, int(radius_data.shape[0] * self.min_cluster_size))
            
            # Run OPTICS
            optics = OPTICS(
                min_samples=self.min_samples,
                xi=self.xi,
                min_cluster_size=min_cluster_size_abs,
                metric='euclidean',
                n_jobs=-1
            )
            
            clusters = optics.fit_predict(radius_data)
            
            # Store in matrix
            col_idx = radius - self.min_radius
            radius_optimal_clusters[:len(clusters), col_idx] = clusters
            
            print(f"  Found {len(np.unique(clusters[clusters != -1]))} clusters (excluding noise)")
            
        return radius_optimal_clusters
        
    def process_clusters(self, radius_optimal_clusters: np.ndarray) -> List:
        """Process clusters to extract TADs"""
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
        """Number the TAD clusters found"""
        length = len(found_tad)
        assign = np.zeros(length, dtype=int)
        count = 1
        assign[0] = count
        
        for i in range(1, length):
            if found_tad[i-1] != found_tad[i]:
                count += 1
            assign[i] = count
            
        return assign