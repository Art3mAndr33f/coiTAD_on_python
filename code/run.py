#!/usr/bin/env python3
"""CLI entry point for coiTAD."""

import argparse, sys


def main():
    p = argparse.ArgumentParser(prog="coiTAD")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("mcool")
    common.add_argument("--chr", default="chr19", dest="chromosome")
    common.add_argument("--res", type=int, default=50000, dest="resolution")
    common.add_argument("--out", default=None, dest="output_dir")

    ps = sub.add_parser("single", parents=[common])
    ps.add_argument("--method", default="HDBSCAN", choices=["HDBSCAN","OPTICS"])
    ps.add_argument("--no-viz", action="store_true")

    sub.add_parser("compare", parents=[common])

    pf = sub.add_parser("full", parents=[common])
    pf.add_argument("--genome", default="hg19")

    args = p.parse_args()

    if args.command == "single":
        from pipeline import run_coitad
        run_coitad(args.mcool, args.chromosome, args.resolution,
                   args.output_dir or "coitad_output", args.method,
                   visualize=not args.no_viz)
    elif args.command == "compare":
        from pipeline import run_comparison
        run_comparison(args.mcool, args.chromosome, args.resolution,
                       args.output_dir or "method_comparison")
    elif args.command == "full":
        from pipeline import run_full_analysis
        run_full_analysis(args.mcool, args.chromosome, args.resolution,
                          args.output_dir or "full_comparison", genome=args.genome)


if __name__ == "__main__":
    main()