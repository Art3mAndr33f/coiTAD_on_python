"""
coiTAD algorithm — HDBSCAN and OPTICS clustering backends.
Original: Oluwatosin Oluwadare / Drew Houchens, UCCS Bioinformatics Lab.
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import hdbscan
from sklearn.cluster import OPTICS

from feature_generation import FeatureGenerator
from extract_tad import ExtractTAD
from quality_check import QualityChecker

warnings.filterwarnings('ignore')


class CoiTADBase(ABC):
    def __init__(self, filepath, feature_filepath, filename,
                 chromo='chr', resolution=50000,
                 max_tad_size=800000, output_folder='data_Results'):
        self.filepath = Path(filepath)
        self.feature_filepath = Path(feature_filepath)
        self.filename = filename
        self.chromo = chromo
        self.resolution = resolution
        self.max_tad_size = max_tad_size
        self.output_folder = output_folder
        self.min_radius = 2
        self.max_radius = int((max_tad_size / resolution) + 10)
        self.max_quality = 0
        self.best_radius = None
        self.chr_data = None
        self.result_path = None

    @property
    @abstractmethod
    def algorithm_name(self) -> str: ...

    @abstractmethod
    def cluster_features(self, feature_data: np.ndarray) -> np.ndarray: ...

    def load_data(self):
        full_path = self.filepath / self.filename
        print(f"Loading data from {full_path}...")
        self.chr_data = np.loadtxt(full_path)
        self.result_path = Path(self.output_folder)
        self.result_path.mkdir(exist_ok=True)
        print("Data set loaded.")

    def run(self):
        self.load_data()
        print("Generating features...")
        feat_dir = Path(self.feature_filepath)
        first_feat = feat_dir / f"feature_radius_{self.min_radius}.txt"
        if not first_feat.exists():
            FeatureGenerator(
                contact_matrix=self.chr_data,
                min_radius=self.min_radius,
                max_radius=self.max_radius,
                output_folder=self.feature_filepath
            ).generate_all_features()
        else:
            print("  Features already exist, skipping generation.")

        print("Performing clustering...")
        radius_clusters = self._perform_clustering()
        print("=" * 60)
        print("Quality Assessment")
        print("=" * 60)
        tad_quality = self._process_clusters(radius_clusters)
        self._quality_check(tad_quality)
        print("=" * 60)
        print(f"Recommended radius = {self.best_radius}")
        print(f"{self.algorithm_name} coiTAD Completed")
        print("=" * 60)

    def _perform_clustering(self):
        max_length = 0
        radii_data = {}
        for radius in range(self.min_radius, self.max_radius + 1):
            fp = Path(self.feature_filepath) / f'feature_radius_{radius}.txt'
            data = np.loadtxt(fp)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            radii_data[radius] = data
            max_length = max(max_length, data.shape[0])
        num_radii = self.max_radius - self.min_radius + 1
        result = np.zeros((max_length, num_radii))
        for radius, data in radii_data.items():
            print(f"Processing radius = {radius}")
            labels = self.cluster_features(data)
            col = radius - self.min_radius
            result[:len(labels), col] = labels
        return result

    def _process_clusters(self, radius_clusters):
        tad_quality = []
        for col in range(radius_clusters.shape[1]):
            radius = self.min_radius + col
            print(f"Result when radius = {radius}")
            clusters = radius_clusters[:, col]
            assign = self._order_tad_num(clusters)
            extractor = ExtractTAD(
                chr_data=self.chr_data, assign_cluster=assign,
                radius=radius, resolution=self.resolution,
                algorithm=self.algorithm_name, result_path=self.result_path)
            tad_quality.append(extractor.extract())
        return tad_quality

    def _quality_check(self, tad_quality):
        quality_path = self.result_path / 'Quality'
        quality_path.mkdir(exist_ok=True)
        checker = QualityChecker(
            chr_data=self.chr_data, resolution=self.resolution,
            min_radius=self.min_radius, max_radius=self.max_radius,
            tad_quality=tad_quality, result_path=self.result_path,
            quality_path=quality_path, algorithm=self.algorithm_name)
        self.best_radius = checker.check()
        self.max_quality = checker.max_quality

    @staticmethod
    def _order_tad_num(found_tad):
        assign = np.zeros(len(found_tad), dtype=int)
        count = 1
        assign[0] = count
        for i in range(1, len(found_tad)):
            if found_tad[i - 1] != found_tad[i]:
                count += 1
            assign[i] = count
        return assign


class CoiTAD_HDBSCAN(CoiTADBase):
    algorithm_name = 'HDBSCAN'

    def cluster_features(self, feature_data):
        clusterer = hdbscan.HDBSCAN(metric='euclidean')
        clusterer.fit(feature_data)
        return clusterer.labels_


class CoiTAD_OPTICS(CoiTADBase):
    algorithm_name = 'OPTICS'

    def __init__(self, *args, min_samples=5, xi=0.05,
                 min_cluster_size=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size_frac = min_cluster_size

    def cluster_features(self, feature_data):
        min_cs = max(2, int(feature_data.shape[0] * self.min_cluster_size_frac))
        optics = OPTICS(min_samples=self.min_samples, xi=self.xi,
                        min_cluster_size=min_cs, metric='euclidean', n_jobs=-1)
        labels = optics.fit_predict(feature_data)
        n_clusters = len(np.unique(labels[labels != -1]))
        print(f"  Found {n_clusters} clusters (excluding noise)")
        return labels