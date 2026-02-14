"""
Script to run comparison between HDBSCAN and OPTICS
Скрипт для запуска сравнения методов
"""

from pathlib import Path


def run_full_comparison(
    mcool_file: str,
    chromosome: str = "chr19",
    resolution: int = 50000,
    output_dir: str = "method_comparison"
):
    """
    Run complete comparison pipeline
    
    Args:
        mcool_file: Path to .mcool file
        chromosome: Chromosome to analyze
        resolution: Resolution in bp
        output_dir: Output directory
    """
    from mcool_converter import McoolConverter
    from coitad_main import CoiTAD
    from coitad_optics import CoiTAD_OPTICS
    from comparison_framework import TADComparator
    
    print("=" * 70)
    print("COITAD METHOD COMPARISON: HDBSCAN vs OPTICS")
    print("=" * 70)
    
    # Create directories
    Path(output_dir).mkdir(exist_ok=True)
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Step 1: Convert mcool to matrix
    print("\nStep 1: Converting .mcool file...")
    converter = McoolConverter(mcool_file)
    
    output_matrix = data_dir / f"{chromosome}_{resolution//1000}kb.hic"
    converter.extract_chromosome(
        chromosome=chromosome,
        resolution=resolution,
        output_file=str(output_matrix),
        balance=True
    )
    
    # Step 2: Run HDBSCAN version
    print("\n" + "=" * 70)
    print("Step 2: Running coiTAD with HDBSCAN...")
    print("=" * 70)
    
    coitad_hdbscan = CoiTAD(
        filepath=str(data_dir),
        feature_filepath=str(Path(output_dir) / "features_hdbscan"),
        filename=f"{chromosome}_{resolution//1000}kb.hic",
        resolution=resolution,
        output_folder=str(Path(output_dir) / "results_hdbscan")
    )
    coitad_hdbscan.run()
    
    # Step 3: Run OPTICS version
    print("\n" + "=" * 70)
    print("Step 3: Running coiTAD with OPTICS...")
    print("=" * 70)
    
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
    
    # Step 4: Compare results
    print("\n" + "=" * 70)
    print("Step 4: Comparing results...")
    print("=" * 70)
    
    comparator = TADComparator(output_dir=str(Path(output_dir) / "comparison"))
    
    # Load TAD results
    hdbscan_tad_file = (Path(output_dir) / "results_hdbscan" / "TADs" / 
                       f"HDBSCAN_{coitad_hdbscan.best_radius}_TAD_BinID.txt")
    optics_tad_file = (Path(output_dir) / "results_optics" / "TADs" / 
                      f"OPTICS_{coitad_optics.best_radius}_TAD_BinID.txt")
    
    method_results = {
        'HDBSCAN': comparator.load_tad_results(str(hdbscan_tad_file)),
        'OPTICS': comparator.load_tad_results(str(optics_tad_file))
    }
    
    # Generate comparisons
    print("\nGenerating comparison plots...")
    comparator.plot_tad_count_comparison(method_results)
    comparator.plot_tad_size_comparison(method_results, resolution)
    comparator.plot_moc_heatmap(method_results)
    
    # Generate report
    print("\nGenerating comparison report...")
    comparator.generate_comparison_report(method_results, resolution)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED!")
    print("=" * 70)
    print(f"Results saved in: {output_dir}/comparison/")
    print(f"\nBest radius HDBSCAN: {coitad_hdbscan.best_radius}")
    print(f"Best radius OPTICS: {coitad_optics.best_radius}")
    

# Example usage
if __name__ == "__main__":
    run_full_comparison(
        mcool_file="4DNFI52OLNJ4.mcool",
        chromosome="chr19",
        resolution=50000,
        output_dir="hdbscan_vs_optics_comparison"
    )