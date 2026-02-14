# run_complete_comparison.py
"""
Complete comparison script with biological validation
Полный скрипт сравнения с биологической валидацией
"""

from pathlib import Path


def run_complete_analysis(
    mcool_file: str = "4DNFI52OLNJ4.mcool",
    chromosome: str = "chr19",
    resolution: int = 50000,
    genome: str = "hg19"
):
    """
    Run complete analysis pipeline
    """
    from integration_with_comparison import run_full_comparison_with_validation
    from download_chipseq_data import prepare_chipseq_data
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  coiTAD: COMPREHENSIVE METHOD COMPARISON                            ║
    ║  HDBSCAN vs OPTICS with Biological Validation                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Download ChIP-Seq data
    print("\n" + "=" * 80)
    print("Preparing ChIP-Seq marker data...")
    print("=" * 80)
    
    marker_files = prepare_chipseq_data(genome=genome, output_dir='chipseq_data')
    
    # Step 2: Run complete comparison
    run_full_comparison_with_validation(
        mcool_file=mcool_file,
        chromosome=chromosome,
        resolution=resolution,
        output_dir="complete_comparison",
        marker_files=marker_files if marker_files else None,
        genome=genome
    )
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE ANALYSIS FINISHED!")
    print("=" * 80)
    print("\nCheck the following directories for results:")
    print("  - complete_comparison/comparison/          : Structural comparison")
    print("  - complete_comparison/biological_validation/: Biological validation")
    print("  - chipseq_data/                            : Downloaded ChIP-Seq data")


if __name__ == "__main__":
    run_complete_analysis()