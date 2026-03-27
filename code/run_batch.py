#!/usr/bin/env python3
"""Batch runner: datasets x chromosomes x resolutions x methods. Skips existing."""

import time, re, numpy as np, pandas as pd
from pathlib import Path
from itertools import product

from config import (DATASETS, GENOME, BATCH_OUTPUT as ROOT,
                    CHROMOSOMES, RESOLUTIONS, METHODS, OPTICS_PARAMS)
from mcool_converter import McoolConverter
from coitad import CoiTAD_HDBSCAN, CoiTAD_OPTICS
from comparison import TADComparator
from validation import BiologicalValidator, prepare_chipseq_data


def find_best_tad_file(res_dir, method):
    tad_dir = res_dir / "TADs"
    if not tad_dir.exists():
        return None, None
    qf = res_dir / "Quality" / "Readme.txt"
    if qf.exists():
        text = qf.read_text()
        for part in text.split():
            if f"{method}_" in part and "_domain" in part:
                r = part.replace(f"{method}_", "").replace("_domain.txt", "")
                f = tad_dir / f"{method}_{r}_TAD_BinID.txt"
                if f.exists():
                    return f, int(r)
    candidates = sorted(tad_dir.glob(f"{method}_*_TAD_BinID.txt"))
    if candidates:
        r = int(candidates[-1].stem.split('_')[1])
        return candidates[-1], r
    return None, None


def run_single(mcool, ds_name, chrom, res, method, exp_dir, chipseq):
    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = exp_dir / f"features_{method}"
    res_dir = exp_dir / f"results_{method}"

    existing_file, existing_radius = find_best_tad_file(res_dir, method)
    if existing_file is not None:
        print(f"  SKIP {ds_name}/{chrom}/{res//1000}kb/{method} (cached r={existing_radius})")
        tads = np.loadtxt(str(existing_file), dtype=int)
        if tads.ndim == 1: tads = tads.reshape(1, -1)
        mf = data_dir / f"{chrom}_{res//1000}kb.hic"
        matrix = np.loadtxt(str(mf))
        quality = 0.0
        qf = res_dir / "Quality" / "Readme.txt"
        if qf.exists():
            m = re.search(r'value\s+([\d.]+)', qf.read_text())
            if m: quality = float(m.group(1))
        return _build(ds_name, chrom, res, method, tads, matrix.shape[0],
                      existing_radius, quality, 0, chipseq), tads

    mf = data_dir / f"{chrom}_{res//1000}kb.hic"
    if not mf.exists():
        matrix = McoolConverter(mcool).extract_chromosome(chrom, res, str(mf), balance=True)
    else:
        matrix = np.loadtxt(str(mf))

    common = dict(filepath=str(data_dir), feature_filepath=str(feat_dir),
                  filename=mf.name, resolution=res, output_folder=str(res_dir))
    t0 = time.time()
    if method == "OPTICS":
        runner = CoiTAD_OPTICS(**common, **OPTICS_PARAMS)
    else:
        runner = CoiTAD_HDBSCAN(**common)
    runner.run()
    elapsed = time.time() - t0

    tad_file = res_dir / "TADs" / f"{method}_{runner.best_radius}_TAD_BinID.txt"
    try:
        tads = np.loadtxt(str(tad_file), dtype=int)
        if tads.ndim == 1: tads = tads.reshape(1, -1)
    except Exception:
        tads = np.empty((0, 2), dtype=int)

    return _build(ds_name, chrom, res, method, tads, matrix.shape[0],
                  runner.best_radius, runner.max_quality, elapsed, chipseq), tads


def _build(dataset, chrom, res, method, tads, n_bins, best_radius, quality, elapsed, chipseq):
    if len(tads) > 0:
        sizes = (tads[:, 1] - tads[:, 0]) * res / 1000
        mean_s, med_s = float(np.mean(sizes)), float(np.median(sizes))
    else:
        mean_s = med_s = 0.0
    result = {
        "dataset": dataset, "chromosome": chrom, "resolution_kb": res // 1000,
        "method": method, "n_tads": len(tads),
        "mean_size_kb": round(mean_s, 1), "median_size_kb": round(med_s, 1),
        "best_radius": best_radius, "quality_score": round(quality, 6),
        "n_bins": n_bins, "time_sec": round(elapsed, 1),
    }
    if chipseq and len(tads) > 0:
        v = BiologicalValidator(res, chrom, GENOME, output_dir="__tmp_val")
        v.load_marker_data(chipseq)
        for mn in v.markers:
            e = v.calculate_boundary_enrichment(tads, mn, n_bins, window=2)
            result[f"avg_{mn}_peaks"] = round(e['avg_peaks_per_bin'], 3)
    return result


def run_batch():
    ROOT.mkdir(exist_ok=True)
    chipseq = prepare_chipseq_data(GENOME, str(ROOT / "chipseq_data"))

    all_results = []
    tads_store = {}
    combos = [(ds, chrom, res, method)
              for ds in DATASETS for chrom in CHROMOSOMES
              for res in RESOLUTIONS for method in METHODS]
    total = len(combos)

    for i, (ds_name, chrom, res, method) in enumerate(combos, 1):
        mcool = DATASETS[ds_name]
        exp_dir = ROOT / ds_name / f"{chrom}_{res // 1000}kb"
        print(f"\n{'='*70}\n[{i}/{total}] {ds_name}/{chrom}/{res//1000}kb/{method}\n{'='*70}")
        result, tads = run_single(mcool, ds_name, chrom, res, method, exp_dir, chipseq)
        all_results.append(result)
        tads_store[(ds_name, chrom, res, method)] = (tads, result['n_bins'])
        pd.DataFrame(all_results).to_csv(ROOT / "all_results.csv", index=False)

    df = pd.DataFrame(all_results)
    df.to_csv(ROOT / "all_results.csv", index=False)

    comparisons = []
    for ds_name in DATASETS:
        for chrom, res in product(CHROMOSOMES, RESOLUTIONS):
            kh = (ds_name, chrom, res, "HDBSCAN")
            ko = (ds_name, chrom, res, "OPTICS")
            if kh in tads_store and ko in tads_store:
                th, nb = tads_store[kh]; to, _ = tads_store[ko]
                if len(th) > 0 and len(to) > 0:
                    c = TADComparator.__new__(TADComparator); c.output_dir = Path(".")
                    comp = {"dataset": ds_name, "chromosome": chrom, "resolution_kb": res//1000,
                            "MoC": round(c.calculate_moc(th, to, 2), 4)}
                    cm = c.calculate_clustering_metrics(th, to, nb)
                    comp["ARI"] = round(cm['Adjusted Rand Index'], 4)
                    comp["NMI"] = round(cm['Normalized Mutual Information'], 4)
                    pr = c.calculate_boundary_precision_recall(to, th, 2)
                    comp.update({k: round(v, 4) for k, v in pr.items()})
                    comparisons.append(comp)

    pd.DataFrame(comparisons).to_csv(ROOT / "all_comparisons.csv", index=False)
    print(f"\nDone: {ROOT / 'all_results.csv'}, {ROOT / 'all_comparisons.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_batch()