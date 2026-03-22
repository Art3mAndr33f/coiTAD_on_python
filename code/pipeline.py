"""
Ready-to-use pipelines for coiTAD.

  • run_coitad          — single method (HDBSCAN by default)
  • run_comparison      — HDBSCAN vs OPTICS structural comparison
  • run_full_analysis   — comparison + biological validation
"""

from pathlib import Path
from typing import Dict, Optional


# ==============================================================
# 1. Basic run
# ==============================================================

def run_coitad(
    mcool_file: str,
    chromosome: str = "chr19",
    resolution: int = 50000,
    output_dir: str = "coitad_output",
    method: str = "HDBSCAN",
    max_tad_size: int = 800000,
    visualize: bool = True,
    # OPTICS-specific
    min_samples: int = 5,
    xi: float = 0.05,
    min_cluster_size: float = 0.05,
):
    """
    Full pipeline: .mcool → features → clustering → TADs → visualization.

    Args:
        mcool_file:     Path to .mcool
        chromosome:     e.g. 'chr19'
        resolution:     bp (50000 = 50 kb)
        output_dir:     root output directory
        method:         'HDBSCAN' or 'OPTICS'
        max_tad_size:   max TAD size in bp
        visualize:      generate plots after TAD calling
        min_samples:    OPTICS param
        xi:             OPTICS param
        min_cluster_size: OPTICS param
    """
    from mcool_converter import McoolConverter
    from coitad import CoiTAD_HDBSCAN, CoiTAD_OPTICS
    from visualization import visualize_coitad_results

    root = Path(output_dir)
    root.mkdir(exist_ok=True)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    feat_dir = root / "features"
    feat_dir.mkdir(exist_ok=True)

    # --- convert mcool ---
    print("=" * 60)
    print("Step 1: Converting .mcool")
    print("=" * 60)

    converter = McoolConverter(mcool_file)
    converter.list_resolutions()
    converter.get_chromosome_info(resolution)

    matrix_file = data_dir / f"{chromosome}_{resolution // 1000}kb.hic"
    converter.extract_chromosome(
        chromosome=chromosome,
        resolution=resolution,
        output_file=str(matrix_file),
        balance=True,
    )

    # --- run coiTAD ---
    print("\n" + "=" * 60)
    print(f"Step 2: Running coiTAD ({method})")
    print("=" * 60)

    common_kw = dict(
        filepath=str(data_dir),
        feature_filepath=str(feat_dir),
        filename=f"{chromosome}_{resolution // 1000}kb.hic",
        resolution=resolution,
        max_tad_size=max_tad_size,
        output_folder=str(root / "results"),
    )

    if method.upper() == "OPTICS":
        runner = CoiTAD_OPTICS(
            **common_kw,
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
        )
    else:
        runner = CoiTAD_HDBSCAN(**common_kw)

    runner.run()

    # --- visualize ---
    if visualize:
        print("\nGenerating visualizations...")
        visualize_coitad_results(
            results_dir=str(root / "results"),
            data_dir=str(data_dir),
            chromosome=chromosome,
            resolution=resolution,
            best_radius=runner.best_radius,
            algorithm=runner.algorithm_name,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Results          : {root / 'results'}")
    print(f"Recommended radius: {runner.best_radius}")
    print("=" * 60)
    return runner


# ==============================================================
# 2. HDBSCAN vs OPTICS comparison
# ==============================================================

def run_comparison(
    mcool_file: str,
    chromosome: str = "chr19",
    resolution: int = 50000,
    output_dir: str = "method_comparison",
):
    """
    Structural comparison: HDBSCAN vs OPTICS.
    Plots: TAD counts, size distributions, MoC heatmap, text report.
    """
    from mcool_converter import McoolConverter
    from coitad import CoiTAD_HDBSCAN, CoiTAD_OPTICS
    from comparison import TADComparator

    root = Path(output_dir)
    root.mkdir(exist_ok=True)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    # convert
    print("=" * 70)
    print("COITAD METHOD COMPARISON: HDBSCAN vs OPTICS")
    print("=" * 70)

    converter = McoolConverter(mcool_file)
    matrix_file = data_dir / f"{chromosome}_{resolution // 1000}kb.hic"
    matrix = converter.extract_chromosome(
        chromosome=chromosome,
        resolution=resolution,
        output_file=str(matrix_file),
        balance=True,
    )
    n_bins = matrix.shape[0]

    # HDBSCAN
    print("\n" + "=" * 70)
    print("Running coiTAD — HDBSCAN")
    print("=" * 70)
    hdb = CoiTAD_HDBSCAN(
        filepath=str(data_dir),
        feature_filepath=str(root / "features_hdbscan"),
        filename=matrix_file.name,
        resolution=resolution,
        output_folder=str(root / "results_hdbscan"),
    )
    hdb.run()

    # OPTICS
    print("\n" + "=" * 70)
    print("Running coiTAD — OPTICS")
    print("=" * 70)
    opt = CoiTAD_OPTICS(
        filepath=str(data_dir),
        feature_filepath=str(root / "features_optics"),
        filename=matrix_file.name,
        resolution=resolution,
        output_folder=str(root / "results_optics"),
        min_samples=5, xi=0.05, min_cluster_size=0.05,
    )
    opt.run()

    # compare
    print("\n" + "=" * 70)
    print("Comparing results...")
    print("=" * 70)

    comp = TADComparator(output_dir=str(root / "comparison"))

    hdb_file = (root / "results_hdbscan" / "TADs" /
                f"HDBSCAN_{hdb.best_radius}_TAD_BinID.txt")
    opt_file = (root / "results_optics" / "TADs" /
                f"OPTICS_{opt.best_radius}_TAD_BinID.txt")

    method_results = {
        'HDBSCAN': comp.load_tad_results(str(hdb_file)),
        'OPTICS':  comp.load_tad_results(str(opt_file)),
    }

    comp.plot_tad_count_comparison(method_results)
    comp.plot_tad_size_comparison(method_results, resolution)
    comp.plot_moc_heatmap(method_results)
    comp.generate_comparison_report(method_results, resolution, n_bins=n_bins)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED!")
    print(f"Results : {root / 'comparison'}")
    print(f"Best radius HDBSCAN : {hdb.best_radius}")
    print(f"Best radius OPTICS  : {opt.best_radius}")
    print("=" * 70)

    return hdb, opt, method_results


# ==============================================================
# 3. Full analysis with biological validation
# ==============================================================

def run_full_analysis(
    mcool_file: str,
    chromosome: str = "chr19",
    resolution: int = 50000,
    output_dir: str = "full_comparison",
    marker_files: Optional[Dict[str, str]] = None,
    genome: str = "hg19",
):
    """
    Complete pipeline: HDBSCAN vs OPTICS + ChIP-Seq biological validation.

    Args:
        mcool_file:    Path to .mcool
        chromosome:    e.g. 'chr19'
        resolution:    bp
        output_dir:    root output directory
        marker_files:  optional dict {marker_name: bed_path}
                       If None — download from UCSC ENCODE.
        genome:        genome build for ENCODE data
    """
    from mcool_converter import McoolConverter
    from coitad import CoiTAD_HDBSCAN, CoiTAD_OPTICS
    from comparison import TADComparator
    from validation import BiologicalValidator, prepare_chipseq_data

    root = Path(output_dir)
    root.mkdir(exist_ok=True)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("COITAD: COMPREHENSIVE COMPARISON WITH BIOLOGICAL VALIDATION")
    print("=" * 80)

    # --- convert ---
    converter = McoolConverter(mcool_file)
    matrix_file = data_dir / f"{chromosome}_{resolution // 1000}kb.hic"
    matrix = converter.extract_chromosome(
        chromosome=chromosome, resolution=resolution,
        output_file=str(matrix_file), balance=True)
    n_bins = matrix.shape[0]

    # --- HDBSCAN ---
    print("\n" + "=" * 80)
    print("Running coiTAD — HDBSCAN")
    print("=" * 80)
    hdb = CoiTAD_HDBSCAN(
        filepath=str(data_dir),
        feature_filepath=str(root / "features_hdbscan"),
        filename=matrix_file.name,
        resolution=resolution,
        output_folder=str(root / "results_hdbscan"))
    hdb.run()

    # --- OPTICS ---
    print("\n" + "=" * 80)
    print("Running coiTAD — OPTICS")
    print("=" * 80)
    opt = CoiTAD_OPTICS(
        filepath=str(data_dir),
        feature_filepath=str(root / "features_optics"),
        filename=matrix_file.name,
        resolution=resolution,
        output_folder=str(root / "results_optics"),
        min_samples=5, xi=0.05, min_cluster_size=0.05)
    opt.run()

    # --- structural comparison ---
    print("\n" + "=" * 80)
    print("Structural comparison")
    print("=" * 80)

    comp = TADComparator(output_dir=str(root / "comparison"))

    hdb_file = (root / "results_hdbscan" / "TADs" /
                f"HDBSCAN_{hdb.best_radius}_TAD_BinID.txt")
    opt_file = (root / "results_optics" / "TADs" /
                f"OPTICS_{opt.best_radius}_TAD_BinID.txt")

    method_tads = {
        'HDBSCAN': comp.load_tad_results(str(hdb_file)),
        'OPTICS':  comp.load_tad_results(str(opt_file)),
    }

    comp.plot_tad_count_comparison(method_tads)
    comp.plot_tad_size_comparison(method_tads, resolution)
    comp.plot_moc_heatmap(method_tads)
    comp.generate_comparison_report(method_tads, resolution, n_bins=n_bins)

    # --- biological validation ---
    print("\n" + "=" * 80)
    print("Biological validation")
    print("=" * 80)

    if marker_files is None:
        print("Downloading ChIP-Seq data from ENCODE...")
        marker_files = prepare_chipseq_data(genome=genome,
                                            output_dir=str(root / "chipseq_data"))

    validator = BiologicalValidator(
        resolution=resolution,
        chromosome=chromosome,
        genome=genome,
        output_dir=str(root / "biological_validation"))

    validator.load_marker_data(marker_files if marker_files else None)

    if not validator.markers:
        print("\nWarning: no ChIP-Seq data loaded — skipping biological validation.")
    else:
        results = validator.compare_methods(method_tads, n_bins)
        validator.plot_enrichment_profiles(method_tads, n_bins)
        validator.plot_peaks_per_bin_comparison(method_tads, n_bins)
        validator.generate_validation_report(results)

    # --- summary ---
    print("\n" + "=" * 80)
    print("COMPLETE ANALYSIS FINISHED!")
    print("=" * 80)
    print(f"Structural comparison : {root / 'comparison'}")
    print(f"Biological validation : {root / 'biological_validation'}")
    print(f"Best radius HDBSCAN  : {hdb.best_radius}")
    print(f"Best radius OPTICS   : {opt.best_radius}")
    if validator.markers:
        print(f"Markers validated    : {', '.join(validator.markers.keys())}")
    print("=" * 80)