#!/usr/bin/env python3
"""
Batch runner: multiple chromosomes × resolutions × methods.
Aggregates results into a single summary table.
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

from mcool_converter import McoolConverter
from coitad import CoiTAD_HDBSCAN, CoiTAD_OPTICS
from comparison import TADComparator
from validation import BiologicalValidator, prepare_chipseq_data


# ======================================================================
# Configuration
# ======================================================================

MCOOL_FILE = "4DNFI52OLNJ4.mcool"
GENOME = "hg19"
ROOT_OUTPUT = Path("batch_results")

CHROMOSOMES = ["chr19", "chr17", "chr22"]       # chr1 опционально (долго)
RESOLUTIONS = [25000, 50000, 100000]
METHODS = ["HDBSCAN", "OPTICS"]

# OPTICS defaults
OPTICS_PARAMS = dict(min_samples=5, xi=0.05, min_cluster_size=0.05)


# ======================================================================
# Single experiment
# ======================================================================

def run_single_experiment(mcool_file, chromosome, resolution, method,
                          output_dir, chipseq_markers=None):
    """Run one (chromosome, resolution, method) experiment."""

    data_dir = output_dir / "data"
    feat_dir = output_dir / f"features_{method}"
    res_dir = output_dir / f"results_{method}"

    data_dir.mkdir(parents=True, exist_ok=True)

    # Convert
    converter = McoolConverter(mcool_file)
    matrix_name = f"{chromosome}_{resolution // 1000}kb.hic"
    matrix_file = data_dir / matrix_name

    if not matrix_file.exists():
        matrix = converter.extract_chromosome(
            chromosome=chromosome,
            resolution=resolution,
            output_file=str(matrix_file),
            balance=True)
    else:
        matrix = np.loadtxt(str(matrix_file))

    n_bins = matrix.shape[0]

    # Run TAD calling
    common = dict(
        filepath=str(data_dir),
        feature_filepath=str(feat_dir),
        filename=matrix_name,
        resolution=resolution,
        output_folder=str(res_dir))

    t0 = time.time()

    if method == "OPTICS":
        runner = CoiTAD_OPTICS(**common, **OPTICS_PARAMS)
    else:
        runner = CoiTAD_HDBSCAN(**common)

    runner.run()
    elapsed = time.time() - t0

    # Load TAD results
    tad_file = (res_dir / "TADs" /
                f"{method}_{runner.best_radius}_TAD_BinID.txt")

    try:
        tads = np.loadtxt(str(tad_file), dtype=int)
        if tads.ndim == 1:
            tads = tads.reshape(1, -1)
    except Exception:
        tads = np.empty((0, 2), dtype=int)

    # TAD statistics
    if len(tads) > 0:
        sizes_kb = (tads[:, 1] - tads[:, 0]) * resolution / 1000
        mean_size = float(np.mean(sizes_kb))
        median_size = float(np.median(sizes_kb))
    else:
        mean_size = median_size = 0.0

    result = {
        "chromosome": chromosome,
        "resolution_kb": resolution // 1000,
        "method": method,
        "n_tads": len(tads),
        "mean_size_kb": round(mean_size, 1),
        "median_size_kb": round(median_size, 1),
        "best_radius": runner.best_radius,
        "quality_score": round(runner.max_quality, 6),
        "n_bins": n_bins,
        "time_sec": round(elapsed, 1),
    }

    # Biological validation (if markers available)
    if chipseq_markers:
        validator = BiologicalValidator(
            resolution=resolution,
            chromosome=chromosome,
            genome=GENOME,
            output_dir=str(output_dir / "validation"))
        validator.load_marker_data(chipseq_markers)

        for marker_name in validator.markers:
            enrichment = validator.calculate_boundary_enrichment(
                tads, marker_name, n_bins, window=2)
            result[f"avg_{marker_name}_peaks"] = round(
                enrichment['avg_peaks_per_bin'], 3)

    return result, tads


# ======================================================================
# Pairwise comparison for same (chromosome, resolution)
# ======================================================================

def compare_pair(tads_hdb, tads_opt, resolution, n_bins):
    """Compare HDBSCAN vs OPTICS for one condition."""
    comp = TADComparator.__new__(TADComparator)
    comp.output_dir = Path(".")  # dummy

    moc = comp.calculate_moc(tads_hdb, tads_opt, tolerance=2)
    cm = comp.calculate_clustering_metrics(tads_hdb, tads_opt, n_bins)
    pr = comp.calculate_boundary_precision_recall(tads_opt, tads_hdb, tolerance=2)

    return {
        "MoC": round(moc, 4),
        "ARI": round(cm['Adjusted Rand Index'], 4),
        "NMI": round(cm['Normalized Mutual Information'], 4),
        "Precision": round(pr['Precision'], 4),
        "Recall": round(pr['Recall'], 4),
        "F1": round(pr['F1-Score'], 4),
    }


# ======================================================================
# Main batch
# ======================================================================

def run_batch():
    ROOT_OUTPUT.mkdir(exist_ok=True)

    # Download ChIP-Seq once
    print("=" * 80)
    print("Downloading ChIP-Seq data...")
    print("=" * 80)
    chipseq_dir = ROOT_OUTPUT / "chipseq_data"
    chipseq_markers = prepare_chipseq_data(
        genome=GENOME, output_dir=str(chipseq_dir))

    all_results = []
    all_comparisons = []
    tads_store = {}  # (chr, res, method) -> tads array

    total = len(CHROMOSOMES) * len(RESOLUTIONS) * len(METHODS)
    counter = 0

    for chrom, res in product(CHROMOSOMES, RESOLUTIONS):
        exp_dir = ROOT_OUTPUT / f"{chrom}_{res // 1000}kb"
        exp_dir.mkdir(exist_ok=True)

        for method in METHODS:
            counter += 1
            print("\n" + "=" * 80)
            print(f"[{counter}/{total}]  {chrom}  {res // 1000}kb  {method}")
            print("=" * 80)

            result, tads = run_single_experiment(
                MCOOL_FILE, chrom, res, method, exp_dir, chipseq_markers)

            all_results.append(result)
            tads_store[(chrom, res, method)] = (tads, result['n_bins'])

        # Pairwise comparison
        key_h = (chrom, res, "HDBSCAN")
        key_o = (chrom, res, "OPTICS")

        if key_h in tads_store and key_o in tads_store:
            tads_h, n_bins = tads_store[key_h]
            tads_o, _ = tads_store[key_o]

            if len(tads_h) > 0 and len(tads_o) > 0:
                comp = compare_pair(tads_h, tads_o, res, n_bins)
                comp.update({
                    "chromosome": chrom,
                    "resolution_kb": res // 1000,
                })
                all_comparisons.append(comp)

    # ---- Save aggregated results ----

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(ROOT_OUTPUT / "all_results.csv", index=False)
    print(f"\nAll results saved: {ROOT_OUTPUT / 'all_results.csv'}")

    df_comp = pd.DataFrame(all_comparisons)
    df_comp.to_csv(ROOT_OUTPUT / "all_comparisons.csv", index=False)
    print(f"Comparisons saved: {ROOT_OUTPUT / 'all_comparisons.csv'}")

    # ---- Print summary tables ----

    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    print(df_results.to_string(index=False))

    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISONS (HDBSCAN vs OPTICS)")
    print("=" * 80)
    print(df_comp.to_string(index=False))

    # ---- Generate aggregated report ----
    generate_aggregate_report(df_results, df_comp)

    return df_results, df_comp


# ======================================================================
# Aggregated report
# ======================================================================

def generate_aggregate_report(df_results, df_comp):
    """Generate the final summary report."""

    lines = [
        "=" * 90,
        "AGGREGATED ANALYSIS REPORT: coiTAD HDBSCAN vs OPTICS",
        "=" * 90, "",
    ]

    # --- Section 1: TAD counts by condition ---
    lines.append("1. TAD COUNTS")
    lines.append("-" * 90)

    pivot_counts = df_results.pivot_table(
        index=['chromosome', 'resolution_kb'],
        columns='method',
        values='n_tads',
        aggfunc='first')
    lines.append(pivot_counts.to_string())
    lines.append("")

    # --- Section 2: Average sizes ---
    lines.append("2. MEAN TAD SIZE (kb)")
    lines.append("-" * 90)

    pivot_sizes = df_results.pivot_table(
        index=['chromosome', 'resolution_kb'],
        columns='method',
        values='mean_size_kb',
        aggfunc='first')
    lines.append(pivot_sizes.to_string())
    lines.append("")

    # --- Section 3: Concordance metrics ---
    lines.append("3. HDBSCAN vs OPTICS CONCORDANCE")
    lines.append("-" * 90)

    if len(df_comp) > 0:
        lines.append(df_comp.to_string(index=False))
    lines.append("")

    # --- Section 4: Biological validation summary ---
    marker_cols = [c for c in df_results.columns if c.startswith("avg_") and c.endswith("_peaks")]

    if marker_cols:
        lines.append("4. BIOLOGICAL VALIDATION (avg peaks/bin at boundaries)")
        lines.append("-" * 90)

        for marker_col in marker_cols:
            marker = marker_col.replace("avg_", "").replace("_peaks", "")
            lines.append(f"\n  {marker}:")
            pivot_m = df_results.pivot_table(
                index=['chromosome', 'resolution_kb'],
                columns='method',
                values=marker_col,
                aggfunc='first')
            lines.append(pivot_m.to_string())
        lines.append("")

    # --- Section 5: Resolution effect ---
    lines.append("5. RESOLUTION EFFECT (averaged across chromosomes)")
    lines.append("-" * 90)

    res_summary = df_results.groupby(['resolution_kb', 'method']).agg({
        'n_tads': 'mean',
        'mean_size_kb': 'mean',
        'quality_score': 'mean',
    }).round(2)
    lines.append(res_summary.to_string())
    lines.append("")

    # --- Section 6: Overall winner ---
    lines.append("6. OVERALL SUMMARY")
    lines.append("-" * 90)

    if marker_cols:
        for method in ["HDBSCAN", "OPTICS"]:
            sub = df_results[df_results['method'] == method]
            avg_enrichment = sub[marker_cols].mean().mean()
            lines.append(f"  {method:10s}: avg enrichment = {avg_enrichment:.3f} peaks/bin")

    lines.append("")
    lines.append("=" * 90)

    report = "\n".join(lines)

    outfile = ROOT_OUTPUT / "aggregate_report.txt"
    outfile.write_text(report)
    print(f"\nAggregate report saved: {outfile}")
    print(report)


if __name__ == "__main__":
    run_batch()