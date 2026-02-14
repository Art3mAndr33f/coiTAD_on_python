# integration_with_comparison.py
"""
Integration of biological validation into comparison framework
Интеграция биологической валидации в framework сравнения
"""

def run_full_comparison_with_validation(
    mcool_file: str,
    chromosome: str = "chr19",
    resolution: int = 50000,
    output_dir: str = "method_comparison",
    marker_files: dict[str, str] = None,
    genome: str = 'hg19'
):
    """
    Run complete comparison with biological validation
    
    Args:
        mcool_file: Path to .mcool file
        chromosome: Chromosome to analyze
        resolution: Resolution in bp
        output_dir: Output directory
        marker_files: Optional dict of custom ChIP-Seq BED files
        genome: Genome build
    """
    from pathlib import Path
    from mcool_converter import McoolConverter
    from coitad_main import CoiTAD
    from coitad_optics import CoiTAD_OPTICS
    from comparison_framework import TADComparator
    from biological_validation import BiologicalValidator
    
    print("=" * 80)
    print("COITAD COMPREHENSIVE COMPARISON: HDBSCAN vs OPTICS")
    print("WITH BIOLOGICAL VALIDATION")
    print("=" * 80)
    
    # Create directories
    Path(output_dir).mkdir(exist_ok=True)
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Step 1: Convert mcool to matrix
    print("\n" + "=" * 80)
    print("Step 1: Converting .mcool file...")
    print("=" * 80)
    
    converter = McoolConverter(mcool_file)
    output_matrix = data_dir / f"{chromosome}_{resolution//1000}kb.hic"
    
    matrix = converter.extract_chromosome(
        chromosome=chromosome,
        resolution=resolution,
        output_file=str(output_matrix),
        balance=True
    )
    
    n_bins = matrix.shape[0]
    
    # Step 2: Run HDBSCAN version
    print("\n" + "=" * 80)
    print("Step 2: Running coiTAD with HDBSCAN...")
    print("=" * 80)
    
    coitad_hdbscan = CoiTAD(
        filepath=str(data_dir),
        feature_filepath=str(Path(output_dir) / "features_hdbscan"),
        filename=f"{chromosome}_{resolution//1000}kb.hic",
        resolution=resolution,
        output_folder=str(Path(output_dir) / "results_hdbscan")
    )
    coitad_hdbscan.run()
    
    # Step 3: Run OPTICS version
    print("\n" + "=" * 80)
    print("Step 3: Running coiTAD with OPTICS...")
    print("=" * 80)
    
    coitad_optics = CoiTAD_OPTICS(
        filepath=str(data_dir),
        feature_filepath=str(Path(output_dir) / "features_optics"),
        filename=f"{chromosome}_{resolution//1000}kb.hic",
        resolution=resolution,
        output_folder=str(Path(output_dir) / "results_optics"),
        min_samples=5,
        xi=0.05,
        min_cluster_size=0.05
    )
    coitad_optics.run()
    
    # Step 4: Compare results (structural)
    print("\n" + "=" * 80)
    print("Step 4: Structural comparison...")
    print("=" * 80)
    
    comparator = TADComparator(output_dir=str(Path(output_dir) / "comparison"))
    
    # Load TAD results
    hdbscan_tad_file = (Path(output_dir) / "results_hdbscan" / "TADs" / 
                       f"HDBSCAN_{coitad_hdbscan.best_radius}_TAD_BinID.txt")
    optics_tad_file = (Path(output_dir) / "results_optics" / "TADs" / 
                      f"OPTICS_{coitad_optics.best_radius}_TAD_BinID.txt")
    
    method_tads = {
        'HDBSCAN': comparator.load_tad_results(str(hdbscan_tad_file)),
        'OPTICS': comparator.load_tad_results(str(optics_tad_file))
    }
    
    # Generate structural comparisons
    print("\nGenerating structural comparison plots...")
    comparator.plot_tad_count_comparison(method_tads)
    comparator.plot_tad_size_comparison(method_tads, resolution)
    comparator.plot_moc_heatmap(method_tads)
    comparator.generate_comparison_report(method_tads, resolution)
    
    # Step 5: Biological validation
    print("\n" + "=" * 80)
    print("Step 5: Biological validation...")
    print("=" * 80)
    
    validator = BiologicalValidator(
        resolution=resolution,
        chromosome=chromosome,
        genome=genome,
        output_dir=str(Path(output_dir) / "biological_validation")
    )
    
    # Load ChIP-Seq data
    print("\nLoading ChIP-Seq marker data...")
    validator.load_marker_data(marker_files)
    
    if len(validator.markers) == 0:
        print("\nWarning: No ChIP-Seq data loaded. Skipping biological validation.")
        print("Please provide marker_files or check UCSC data availability.")
    else:
        # Run validation
        print("\nRunning biological validation...")
        validation_results = validator.compare_methods(method_tads, n_bins)
        
        # Generate validation plots
        print("\nGenerating validation plots...")
        validator.plot_enrichment_profiles(method_tads, n_bins)
        validator.plot_peaks_per_bin_comparison(method_tads, n_bins)
        
        # Generate validation report
        print("\nGenerating validation report...")
        validator.generate_validation_report(validation_results)
    
    # Final summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved in: {output_dir}/")
    print(f"  - Structural comparison: {output_dir}/comparison/")
    print(f"  - Biological validation: {output_dir}/biological_validation/")
    print(f"\nBest radius HDBSCAN: {coitad_hdbscan.best_radius}")
    print(f"Best radius OPTICS: {coitad_optics.best_radius}")
    
    if len(validator.markers) > 0:
        print(f"\nBiological markers validated: {', '.join(validator.markers.keys())}")


# Example with custom ChIP-Seq files
if __name__ == "__main__":
    # Example 1: Using UCSC ENCODE data (automatic download)
    run_full_comparison_with_validation(
        mcool_file="4DNFI52OLNJ4.mcool",
        chromosome="chr19",
        resolution=50000,
        output_dir="full_comparison_with_validation",
        marker_files=None,  # Will try to download from UCSC
        genome='hg19'
    )
    
    # Example 2: Using custom ChIP-Seq BED files
    # run_full_comparison_with_validation(
    #     mcool_file="4DNFI52OLNJ4.mcool",
    #     chromosome="chr19",
    #     resolution=50000,
    #     output_dir="full_comparison_custom_markers",
    #     marker_files={
    #         'CTCF': 'path/to/CTCF_peaks.bed',
    #         'H3K4me1': 'path/to/H3K4me1_peaks.bed',
    #         'H3K27ac': 'path/to/H3K27ac_peaks.bed',
    #         'H3K4me3': 'path/to/H3K4me3_peaks.bed',
    #         'RNAPII': 'path/to/RNAPII_peaks.bed'
    #     },
    #     genome='hg19'
    # )
