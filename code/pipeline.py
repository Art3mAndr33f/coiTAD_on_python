"""Pipelines: run_coitad / run_comparison / run_full_analysis."""

from pathlib import Path
from typing import Dict, Optional


def run_coitad(mcool_file, chromosome="chr19", resolution=50000,
               output_dir="coitad_output", method="HDBSCAN",
               max_tad_size=800000, visualize=True,
               min_samples=5, xi=0.05, min_cluster_size=0.05):
    from mcool_converter import McoolConverter
    from coitad import CoiTAD_HDBSCAN, CoiTAD_OPTICS
    from visualization import visualize_coitad_results

    root = Path(output_dir); root.mkdir(exist_ok=True)
    data_dir = root / "data"; data_dir.mkdir(exist_ok=True)

    matrix_file = data_dir / f"{chromosome}_{resolution//1000}kb.hic"
    if not matrix_file.exists():
        McoolConverter(mcool_file).extract_chromosome(
            chromosome, resolution, str(matrix_file), balance=True)

    common = dict(filepath=str(data_dir),
                  feature_filepath=str(root/"features"),
                  filename=matrix_file.name, resolution=resolution,
                  max_tad_size=max_tad_size, output_folder=str(root/"results"))

    if method.upper() == "OPTICS":
        runner = CoiTAD_OPTICS(**common, min_samples=min_samples,
                               xi=xi, min_cluster_size=min_cluster_size)
    else:
        runner = CoiTAD_HDBSCAN(**common)
    runner.run()

    if visualize:
        visualize_coitad_results(str(root/"results"), str(data_dir),
                                 chromosome, resolution, runner.best_radius,
                                 runner.algorithm_name)
    return runner


def run_comparison(mcool_file, chromosome="chr19", resolution=50000,
                   output_dir="method_comparison"):
    from mcool_converter import McoolConverter
    from coitad import CoiTAD_HDBSCAN, CoiTAD_OPTICS
    from comparison import TADComparator
    import numpy as np

    root = Path(output_dir); root.mkdir(exist_ok=True)
    data_dir = root / "data"; data_dir.mkdir(exist_ok=True)

    matrix_file = data_dir / f"{chromosome}_{resolution//1000}kb.hic"
    if not matrix_file.exists():
        matrix = McoolConverter(mcool_file).extract_chromosome(
            chromosome, resolution, str(matrix_file), balance=True)
    else:
        matrix = np.loadtxt(str(matrix_file))
    n_bins = matrix.shape[0]

    runners = {}
    for Method, Cls, folder in [("HDBSCAN", CoiTAD_HDBSCAN, "results_hdbscan"),
                                 ("OPTICS", CoiTAD_OPTICS, "results_optics")]:
        r = Cls(filepath=str(data_dir),
                feature_filepath=str(root/f"features_{Method.lower()}"),
                filename=matrix_file.name, resolution=resolution,
                output_folder=str(root/folder))
        r.run(); runners[Method] = r

    comp = TADComparator(str(root/"comparison"))
    mr = {}
    for m, r in runners.items():
        f = root/f"results_{m.lower()}"/"TADs"/f"{m}_{r.best_radius}_TAD_BinID.txt"
        mr[m] = comp.load_tad_results(str(f))
    comp.plot_tad_count_comparison(mr)
    comp.plot_tad_size_comparison(mr, resolution)
    comp.plot_moc_heatmap(mr)
    comp.generate_comparison_report(mr, resolution, n_bins)
    return runners, mr


def run_full_analysis(mcool_file, chromosome="chr19", resolution=50000,
                      output_dir="full_comparison",
                      marker_files=None, genome="hg19"):
    from validation import BiologicalValidator, prepare_chipseq_data
    import numpy as np

    runners, method_tads = run_comparison(mcool_file, chromosome, resolution, output_dir)

    root = Path(output_dir)
    matrix_file = root/"data"/f"{chromosome}_{resolution//1000}kb.hic"
    n_bins = np.loadtxt(str(matrix_file)).shape[0]

    if marker_files is None:
        marker_files = prepare_chipseq_data(genome, str(root/"chipseq_data"))

    v = BiologicalValidator(resolution, chromosome, genome, str(root/"biological_validation"))
    v.load_marker_data(marker_files)
    if v.markers:
        res = v.compare_methods(method_tads, n_bins)
        v.plot_enrichment_profiles(method_tads, n_bins)
        v.plot_peaks_per_bin_comparison(method_tads, n_bins)
        v.generate_validation_report(res)