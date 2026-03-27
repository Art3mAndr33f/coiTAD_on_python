#!/usr/bin/env python3
"""
OPTICS hyperparameter tuning.
  --mode simple : tune on one chromosome (fast)
  --mode cv     : tune + cross-chromosome/dataset validation (robust)
Skips existing results automatically.
"""

import argparse, time, re
import numpy as np, pandas as pd
from pathlib import Path
from itertools import product as iprod

from config import (DATASETS, PRIMARY_DATASET, GENOME,
                    TUNE_CHROM, VAL_CHROMS, TUNE_TOP_K as TOP_K,
                    TUNE_RESOLUTION as RESOLUTION, PARAM_GRID,
                    TUNE_SIMPLE_OUTPUT, TUNE_CV_OUTPUT)
from mcool_converter import McoolConverter
from feature_generation import FeatureGenerator
from coitad import CoiTAD_OPTICS
from validation import BiologicalValidator, prepare_chipseq_data


# ======================================================================
# Helpers
# ======================================================================

def prepare_chrom(chrom, output_root, mcool_file):
    data_dir = output_root / f"data_{chrom}"
    feat_dir = output_root / f"features_{chrom}"
    data_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    matrix_name = f"{chrom}_{RESOLUTION//1000}kb.hic"
    matrix_file = data_dir / matrix_name
    if matrix_file.exists():
        print(f"  Matrix {chrom} exists, loading.")
        matrix = np.loadtxt(str(matrix_file))
    else:
        matrix = McoolConverter(mcool_file).extract_chromosome(
            chrom, RESOLUTION, str(matrix_file), balance=True)
    first_feat = feat_dir / "feature_radius_2.txt"
    if not first_feat.exists():
        mr, mxr = 2, int((800000/RESOLUTION)+10)
        FeatureGenerator(matrix, mr, mxr, feat_dir).generate_all_features()
    else:
        print(f"  Features {chrom} exist, skipping.")
    return matrix, matrix_name, data_dir, feat_dir


def run_single(matrix, matrix_name, data_dir, feat_dir, chipseq,
               chrom, ms, xi, mcs, run_dir):
    n_bins = matrix.shape[0]
    quality_file = run_dir / "Quality" / "Readme.txt"
    if quality_file.exists():
        print(f"    CACHED")
        qt = quality_file.read_text(); quality = 0.0
        m = re.search(r'value\s+([\d.]+)', qt)
        if m: quality = float(m.group(1))
        best_r = 2
        for part in qt.split():
            if "OPTICS_" in part and "_domain" in part:
                best_r = int(part.replace("OPTICS_","").replace("_domain.txt",""))
        tad_file = run_dir / "TADs" / f"OPTICS_{best_r}_TAD_BinID.txt"
        try:
            tads = np.loadtxt(str(tad_file), dtype=int)
            if tads.ndim == 1: tads = tads.reshape(1,-1)
        except: tads = np.empty((0,2), dtype=int)
        return _build(chrom, ms, xi, mcs, tads, n_bins, best_r, quality, 0, chipseq)

    t0 = time.time()
    try:
        runner = CoiTAD_OPTICS(
            filepath=str(data_dir), feature_filepath=str(feat_dir),
            filename=matrix_name, resolution=RESOLUTION,
            output_folder=str(run_dir), min_samples=ms, xi=xi, min_cluster_size=mcs)
        runner.run()
        elapsed = time.time()-t0
        tf = run_dir/"TADs"/f"OPTICS_{runner.best_radius}_TAD_BinID.txt"
        tads = np.loadtxt(str(tf), dtype=int)
        if tads.ndim == 1: tads = tads.reshape(1,-1)
        return _build(chrom, ms, xi, mcs, tads, n_bins,
                      runner.best_radius, runner.max_quality, elapsed, chipseq)
    except Exception as e:
        return {"chromosome":chrom,"min_samples":ms,"xi":xi,"min_cluster_size":mcs,
                "n_tads":0,"mean_size_kb":0,"quality_score":0,"best_radius":-1,
                "time_sec":round(time.time()-t0,1),"status":f"ERROR: {e}"}


def _build(chrom, ms, xi, mcs, tads, n_bins, best_r, quality, elapsed, chipseq):
    if len(tads) > 0:
        sizes = (tads[:,1]-tads[:,0])*RESOLUTION/1000
        mean_s, med_s = float(np.mean(sizes)), float(np.median(sizes))
    else: mean_s = med_s = 0.0
    r = {"chromosome":chrom,"min_samples":ms,"xi":xi,"min_cluster_size":mcs,
         "n_tads":len(tads),"mean_size_kb":round(mean_s,1),"median_size_kb":round(med_s,1),
         "best_radius":best_r,"quality_score":round(quality,6),
         "time_sec":round(elapsed,1),"status":"OK"}
    if chipseq and len(tads) > 0:
        v = BiologicalValidator(RESOLUTION, chrom, GENOME, output_dir="__tmp")
        v.load_marker_data(chipseq)
        vals = []
        for mn in v.markers:
            e = v.calculate_boundary_enrichment(tads, mn, n_bins, window=2)
            r[f"avg_{mn}_peaks"] = round(e['avg_peaks_per_bin'],3)
            vals.append(e['avg_peaks_per_bin'])
        r["avg_enrichment_all"] = round(float(np.mean(vals)),3) if vals else 0.0
    return r


# ======================================================================
# Report
# ======================================================================

def generate_report(df, output_dir, title="TUNING REPORT"):
    df_ok = df[(df['status']=='OK') & (df['quality_score']>0)].copy()
    lines = ["="*90, title, "="*90, "",
             f"Total: {len(df)}, quality>0: {len(df_ok)}", ""]
    if len(df_ok) == 0:
        lines.append("WARNING: No runs with quality > 0")
        (output_dir/"tuning_report.txt").write_text("\n".join(lines), encoding='utf-8')
        print("\n".join(lines)); return

    lines += ["TOP 10 BY QUALITY", "-"*90]
    lines.append(df_ok.nlargest(10,'quality_score')[
        ['min_samples','xi','min_cluster_size','n_tads','quality_score','mean_size_kb']
    ].to_string(index=False)); lines.append("")

    if 'avg_enrichment_all' in df_ok.columns:
        lines += ["TOP 10 BY ENRICHMENT", "-"*90]
        mc = [c for c in df_ok.columns if c.startswith('avg_') and c.endswith('_peaks')]
        lines.append(df_ok.nlargest(10,'avg_enrichment_all')[
            ['min_samples','xi','min_cluster_size','n_tads']+mc+['avg_enrichment_all']
        ].to_string(index=False)); lines.append("")

    df_ok['rq'] = df_ok['quality_score'].rank(ascending=False)
    df_ok['rb'] = df_ok['avg_enrichment_all'].rank(ascending=False) if 'avg_enrichment_all' in df_ok.columns else 1
    df_ok['rs'] = abs(df_ok['mean_size_kb']-800).rank(ascending=True)
    df_ok['combined'] = 0.30*df_ok['rq'] + 0.35*df_ok['rb'] + 0.20*df_ok['rs']
    df_ok['rank'] = df_ok['combined'].rank(ascending=True).astype(int)
    top = df_ok.sort_values('rank')
    lines += ["TOP 15 COMBINED", "-"*90]
    cols = ['min_samples','xi','min_cluster_size','n_tads','quality_score','mean_size_kb']
    if 'avg_enrichment_all' in top.columns: cols.append('avg_enrichment_all')
    cols.append('rank')
    lines.append(top.head(15)[[c for c in cols if c in top.columns]].to_string(index=False))
    best = top.iloc[0]
    lines += ["","="*90, "RECOMMENDED:",
              f"  min_samples={int(best['min_samples'])}  xi={best['xi']}  mcs={best['min_cluster_size']}",
              f"  TADs={int(best['n_tads'])}  quality={best['quality_score']:.6f}  size={best['mean_size_kb']:.1f}kb"]
    if 'avg_enrichment_all' in best.index:
        lines.append(f"  enrichment={best['avg_enrichment_all']:.3f}")
    lines += ["="*90,"","SENSITIVITY","-"*90]
    for p in ['min_samples','xi']:
        g = df_ok.groupby(p).agg({'n_tads':'mean','quality_score':'mean','mean_size_kb':'mean'}).round(3)
        if 'avg_enrichment_all' in df_ok.columns:
            g['enrichment'] = df_ok.groupby(p)['avg_enrichment_all'].mean().round(3)
        lines.append(f"\n  {p}:"); lines.append("  "+g.to_string().replace("\n","\n  "))
    text = "\n".join(lines)
    (output_dir/"tuning_report.txt").write_text(text, encoding='utf-8')
    print(text)


# ======================================================================
# Mode: simple
# ======================================================================

def run_simple(output_dir):
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "tuning_results.csv"
    if csv_path.exists():
        print(f"Found {csv_path} - regenerating report only.")
        generate_report(pd.read_csv(csv_path), output_dir,
                        f"TUNING REPORT ({TUNE_CHROM} {RESOLUTION//1000}kb)")
        return

    chipseq = prepare_chipseq_data(GENOME, str(output_dir/"chipseq_data"))
    mcool = DATASETS[PRIMARY_DATASET]
    matrix, mname, ddir, fdir = prepare_chrom(TUNE_CHROM, output_dir, mcool)

    params = list(iprod(PARAM_GRID["min_samples"], PARAM_GRID["xi"], PARAM_GRID["min_cluster_size"]))
    results = []
    for i, (ms, xi, mcs) in enumerate(params, 1):
        print(f"\n[{i}/{len(params)}] ms={ms} xi={xi} mcs={mcs}")
        r = run_single(matrix, mname, ddir, fdir, chipseq, TUNE_CHROM,
                       ms, xi, mcs, output_dir/f"run_{i}")
        results.append(r)
        pd.DataFrame(results).to_csv(csv_path, index=False)
    generate_report(pd.DataFrame(results), output_dir,
                    f"TUNING REPORT ({TUNE_CHROM} {RESOLUTION//1000}kb)")


# ======================================================================
# Mode: cross-validation
# ======================================================================

def run_cv(output_dir):
    output_dir.mkdir(exist_ok=True)
    p1_csv = output_dir / "phase1_results.csv"
    p2_csv = output_dir / "phase2_results.csv"
    agg_csv = output_dir / "aggregated_rankings.csv"
    chipseq = prepare_chipseq_data(GENOME, str(output_dir/"chipseq_data"))

    # Phase 1: broad search on primary dataset / tune chromosome
    if p1_csv.exists():
        print(f"Phase 1 cached: {p1_csv}")
        df1 = pd.read_csv(p1_csv)
    else:
        print("\n"+"="*80+f"\nPHASE 1: Tuning on {PRIMARY_DATASET}/{TUNE_CHROM}\n"+"="*80)
        mcool = DATASETS[PRIMARY_DATASET]
        matrix, mname, ddir, fdir = prepare_chrom(TUNE_CHROM, output_dir, mcool)
        params = list(iprod(PARAM_GRID["min_samples"], PARAM_GRID["xi"], PARAM_GRID["min_cluster_size"]))
        results = []
        for i, (ms, xi, mcs) in enumerate(params, 1):
            print(f"\n[{i}/{len(params)}] ms={ms} xi={xi} mcs={mcs}")
            r = run_single(matrix, mname, ddir, fdir, chipseq, TUNE_CHROM,
                           ms, xi, mcs, output_dir/f"run_p1_{i}")
            results.append(r)
            pd.DataFrame(results).to_csv(p1_csv, index=False)
        df1 = pd.DataFrame(results); df1.to_csv(p1_csv, index=False)

    # Select top-K
    df1_ok = df1[(df1['status']=='OK') & (df1['quality_score']>0)]
    if len(df1_ok) < TOP_K:
        df1_backup = df1[df1['status']=='OK']
        if 'avg_enrichment_all' in df1_backup.columns:
            df1_backup = df1_backup.nlargest(TOP_K, 'avg_enrichment_all')
        df1_ok = pd.concat([df1_ok, df1_backup]).drop_duplicates(
            subset=['min_samples','xi','min_cluster_size'])
    top_configs = df1_ok.nlargest(TOP_K, 'quality_score')[
        ['min_samples','xi','min_cluster_size']].drop_duplicates()
    print(f"\nTop {len(top_configs)} configs for Phase 2")

    # Phase 2: validate on other chromosomes x other datasets
    if p2_csv.exists():
        print(f"Phase 2 cached: {p2_csv}")
        df2 = pd.read_csv(p2_csv)
    else:
        print("\n"+"="*80+"\nPHASE 2: Cross-dataset/chromosome validation\n"+"="*80)

        VAL_CONDITIONS = []
        for ds_name, mcool in DATASETS.items():
            for chrom in VAL_CHROMS:
                VAL_CONDITIONS.append((ds_name, chrom, mcool))
            if ds_name != PRIMARY_DATASET:
                VAL_CONDITIONS.append((ds_name, TUNE_CHROM, mcool))

        chrom_data = {}
        for ds_name, chrom, mcool in VAL_CONDITIONS:
            key = (ds_name, chrom)
            if key not in chrom_data:
                sub_dir = output_dir / ds_name
                sub_dir.mkdir(exist_ok=True)
                chrom_data[key] = prepare_chrom(chrom, sub_dir, mcool)

        results = []
        total = len(top_configs) * len(VAL_CONDITIONS); cnt = 0
        for _, cfg in top_configs.iterrows():
            ms, xi, mcs = int(cfg['min_samples']), float(cfg['xi']), float(cfg['min_cluster_size'])
            for ds_name, chrom, mcool in VAL_CONDITIONS:
                cnt += 1
                print(f"\n[{cnt}/{total}] {ds_name}/{chrom} ms={ms} xi={xi}")
                matrix, mname, ddir, fdir = chrom_data[(ds_name, chrom)]
                run_dir = output_dir / f"run_p2_{ds_name}_{chrom}_{cnt}"
                r = run_single(matrix, mname, ddir, fdir, chipseq, chrom,
                               ms, xi, mcs, run_dir)
                r['dataset'] = ds_name
                results.append(r)
                pd.DataFrame(results).to_csv(p2_csv, index=False)
        df2 = pd.DataFrame(results); df2.to_csv(p2_csv, index=False)

    # Phase 3: aggregate
    print("\n"+"="*80+"\nPHASE 3: Aggregation\n"+"="*80)
    df1_tagged = df1.copy(); df1_tagged['dataset'] = PRIMARY_DATASET
    df_all = pd.concat([df1_tagged, df2], ignore_index=True)
    df_all = df_all[df_all['status']=='OK'].copy()
    df_all['config'] = (df_all['min_samples'].astype(str)+'_'+
                        df_all['xi'].astype(str)+'_'+
                        df_all['min_cluster_size'].astype(str))

    agg_cols = {'n_tads':'mean','quality_score':'mean','mean_size_kb':'mean','time_sec':'sum'}
    if 'avg_enrichment_all' in df_all.columns: agg_cols['avg_enrichment_all'] = 'mean'
    df_agg = df_all.groupby('config').agg(agg_cols).reset_index()
    qstd = df_all.groupby('config')['quality_score'].std().reset_index()
    qstd.columns = ['config','quality_std']
    df_agg = df_agg.merge(qstd, on='config')
    qpos = df_all[df_all['quality_score']>0].groupby('config').size().reset_index()
    qpos.columns = ['config','n_conditions_pos']
    df_agg = df_agg.merge(qpos, on='config', how='left')
    df_agg['n_conditions_pos'] = df_agg['n_conditions_pos'].fillna(0).astype(int)
    df_agg[['min_samples','xi','min_cluster_size']] = df_agg['config'].str.split('_',expand=True).astype(float)

    df_v = df_agg[df_agg['quality_score']>0].copy()
    if len(df_v) == 0: df_v = df_agg.nlargest(10,'quality_score').copy()
    df_v['rq'] = df_v['quality_score'].rank(ascending=False)
    df_v['rstab'] = df_v['quality_std'].rank(ascending=True)
    df_v['rb'] = df_v['avg_enrichment_all'].rank(ascending=False) if 'avg_enrichment_all' in df_v.columns else 1
    df_v['rs'] = abs(df_v['mean_size_kb']-800).rank(ascending=True)
    df_v['combined'] = 0.25*df_v['rq']+0.30*df_v['rb']+0.25*df_v['rstab']+0.20*df_v['rs']
    df_v['final_rank'] = df_v['combined'].rank(ascending=True).astype(int)
    df_v = df_v.sort_values('final_rank')
    df_v.to_csv(agg_csv, index=False)

    best = df_v.iloc[0]
    lines = ["="*95,"CROSS-VALIDATION TUNING REPORT",
             f"Tune: {PRIMARY_DATASET}/{TUNE_CHROM}",
             f"Validate: {VAL_CHROMS} x {list(DATASETS.keys())}","="*95,""]
    top_cfgs = df_v.head(10)['config'].tolist()
    per = df_all[df_all['config'].isin(top_cfgs)]
    for metric in ['quality_score','n_tads']:
        piv = per.pivot_table(index='config',columns=['dataset','chromosome'],values=metric,aggfunc='first')
        lines += [f"\n{metric}:",piv.to_string(),""]
    cols = ['min_samples','xi','min_cluster_size','n_tads','quality_score',
            'quality_std','mean_size_kb','n_conditions_pos']
    if 'avg_enrichment_all' in df_v.columns: cols.append('avg_enrichment_all')
    cols.append('final_rank')
    lines += ["COMBINED RANKING","-"*95,
              df_v.head(15)[[c for c in cols if c in df_v.columns]].to_string(index=False),""]
    lines += ["="*95,"RECOMMENDED:",
              f"  min_samples={int(best['min_samples'])}  xi={best['xi']}  mcs={best['min_cluster_size']}",
              f"  quality={best['quality_score']:.6f}  std={best['quality_std']:.6f}",
              f"  TADs={best['n_tads']:.0f}  size={best['mean_size_kb']:.1f}kb"]
    if 'avg_enrichment_all' in best.index:
        lines.append(f"  enrichment={best['avg_enrichment_all']:.3f}")
    lines.append("="*95)
    text = "\n".join(lines)
    (output_dir/"tuning_cv_report.txt").write_text(text, encoding='utf-8')
    print(text)


# ======================================================================
# Main
# ======================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="simple", choices=["simple","cv"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.mode == "simple":
        run_simple(Path(args.out) if args.out else TUNE_SIMPLE_OUTPUT)
    else:
        run_cv(Path(args.out) if args.out else TUNE_CV_OUTPUT)


if __name__ == "__main__":
    main()