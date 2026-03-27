"""
Central configuration for coiTAD project.
Edit this file to change datasets, chromosomes, resolutions, etc.
All scripts import from here — no duplication.
"""

from pathlib import Path

# ======================================================================
# Datasets
# ======================================================================

DATASETS = {
    "H1_DpnII": "data/4DNFI52OLNJ4.mcool",
    "H1_MboI":  "data/4DNFIHFX73VQ.mcool",
}

PRIMARY_DATASET = "H1_DpnII"

# ======================================================================
# Analysis parameters
# ======================================================================

GENOME = "hg19"
CHROMOSOMES = ["chr19", "chr17", "chr22"]
RESOLUTIONS = [25000, 50000, 100000]
METHODS = ["HDBSCAN", "OPTICS"]

# OPTICS parameters (update after tuning)
OPTICS_PARAMS = dict(min_samples=15, xi=0.05, min_cluster_size=0.05)

MAX_TAD_SIZE = 800000

# ======================================================================
# Tuning
# ======================================================================

TUNE_CHROM = "chr19"
VAL_CHROMS = ["chr17", "chr22"]
TUNE_TOP_K = 10
TUNE_RESOLUTION = 50000

PARAM_GRID = {
    "min_samples":     [10, 15, 20, 25, 30, 40, 50],
    "xi":              [0.01, 0.03, 0.05, 0.07, 0.10],
    "min_cluster_size": [0.05],
}

# ======================================================================
# Output directories
# ======================================================================

BATCH_OUTPUT = Path("batch_results")
TUNE_SIMPLE_OUTPUT = Path("tuning_results")
TUNE_CV_OUTPUT = Path("tuning_cv_results")

# ======================================================================
# Visualization
# ======================================================================

COLORS = {"HDBSCAN": "blue", "OPTICS": "red"}