# advanced_comparison.py
"""
Advanced comparison with multiple metrics
Продвинутое сравнение с множественными метриками
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class AdvancedComparator:
    """Advanced TAD comparison with clustering metrics"""
    
    def __init__(self, contact_matrix: np.ndarray, resolution: int):
        self.contact_matrix = contact_matrix
        self.resolution = resolution
        
    def tads_to_labels(self, tads: np.ndarray, n_bins: int) -> np.ndarray:
        """Convert TAD boundaries to bin labels"""
        labels = np.zeros(n_bins, dtype=int)
        
        for tad_id, (start, end) in enumerate(tads, start=1):
            labels[start:end+1] = tad_id
            
        return labels
    
    def calculate_clustering_metrics(self, tads1: np.ndarray, tads2: np.ndarray) -> Dict:
        """Calculate clustering quality metrics"""
        n_bins = self.contact_matrix.shape[0]
        
        labels1 = self.tads_to_labels(tads1, n_bins)
        labels2 = self.tads_to_labels(tads2, n_bins)
        
        metrics = {
            'Adjusted Rand Index': adjusted_rand_score(labels1, labels2),
            'Normalized Mutual Information': normalized_mutual_info_score(labels1, labels2)
        }
        
        return metrics
    
    def calculate_boundary_precision_recall(self, pred_tads: np.ndarray, 
                                           true_tads: np.ndarray,
                                           tolerance: int = 2) -> Dict:
        """Calculate precision and recall for boundary detection"""
        pred_boundaries = set()
        for tad in pred_tads:
            pred_boundaries.add(tad[0])
            pred_boundaries.add(tad[1])
        
        true_boundaries = set()
        for tad in true_tads:
            true_boundaries.add(tad[0])
            true_boundaries.add(tad[1])
        
        # True positives
        tp = 0
        for pred_b in pred_boundaries:
            for true_b in true_boundaries:
                if abs(pred_b - true_b) <= tolerance:
                    tp += 1
                    break
        
        precision = tp / len(pred_boundaries) if len(pred_boundaries) > 0 else 0
        recall = tp / len(true_boundaries) if len(true_boundaries) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }


# Example comprehensive comparison
def comprehensive_comparison():
    """Run comprehensive comparison pipeline"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   coiTAD: HDBSCAN vs OPTICS Comprehensive Comparison      ║
    ╚════════════════════════════════════════════════════════════╝
    
    This script will:
    1. Run coiTAD with both HDBSCAN and OPTICS
    2. Compare TAD counts and sizes
    3. Calculate Measure of Concordance (MoC)
    4. Generate visualizations
    5. Produce comparison report
    
    """)
    
    run_full_comparison(
        mcool_file="4DNFI52OLNJ4.mcool",
        chromosome="chr19",
        resolution=50000
    )


if __name__ == "__main__":
    comprehensive_comparison()
