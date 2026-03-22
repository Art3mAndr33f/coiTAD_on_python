#!/usr/bin/env python3
"""
coiTAD — entry point.

Usage examples
--------------

# 1) Single method (HDBSCAN, default)
python run.py single  data.mcool --chr chr19 --res 50000

# 2) Single method (OPTICS)
python run.py single  data.mcool --chr chr19 --res 50000 --method OPTICS

# 3) HDBSCAN vs OPTICS comparison
python run.py compare data.mcool --chr chr19 --res 50000

# 4) Full analysis with biological validation
python run.py full    data.mcool --chr chr19 --res 50000 --genome hg19
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        prog="coiTAD",
        description="TAD calling with circle-of-influence features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- shared arguments ----
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("mcool", help="Path to .mcool file")
    common.add_argument("--chr", default="chr19", dest="chromosome",
                        help="Chromosome (default: chr19)")
    common.add_argument("--res", type=int, default=50000, dest="resolution",
                        help="Resolution in bp (default: 50000)")
    common.add_argument("--out", default=None, dest="output_dir",
                        help="Output directory")

    # ---- single ----
    p_single = sub.add_parser("single", parents=[common],
                              help="Run single method")
    p_single.add_argument("--method", default="HDBSCAN",
                          choices=["HDBSCAN", "OPTICS"],
                          help="Clustering method (default: HDBSCAN)")
    p_single.add_argument("--max-tad-size", type=int, default=800000)
    p_single.add_argument("--no-viz", action="store_true",
                          help="Skip visualization")

    # ---- compare ----
    sub.add_parser("compare", parents=[common],
                   help="HDBSCAN vs OPTICS comparison")

    # ---- full ----
    p_full = sub.add_parser("full", parents=[common],
                            help="Comparison + biological validation")
    p_full.add_argument("--genome", default="hg19",
                        help="Genome build (default: hg19)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "single":
        from pipeline import run_coitad
        run_coitad(
            mcool_file=args.mcool,
            chromosome=args.chromosome,
            resolution=args.resolution,
            output_dir=args.output_dir or "coitad_output",
            method=args.method,
            max_tad_size=args.max_tad_size,
            visualize=not args.no_viz,
        )

    elif args.command == "compare":
        from pipeline import run_comparison
        run_comparison(
            mcool_file=args.mcool,
            chromosome=args.chromosome,
            resolution=args.resolution,
            output_dir=args.output_dir or "method_comparison",
        )

    elif args.command == "full":
        from pipeline import run_full_analysis
        run_full_analysis(
            mcool_file=args.mcool,
            chromosome=args.chromosome,
            resolution=args.resolution,
            output_dir=args.output_dir or "full_comparison",
            genome=args.genome,
        )

    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()