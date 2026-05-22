#!/usr/bin/env python3
"""Harvest LSAP results: per-cell AUC (val_loss-best + val_AUC-best) + vanilla comparison."""
import os, glob, pickle, re
import numpy as np
from collections import defaultdict

# Vanilla AUCs from ECC_DI_REPORT.md (5-seed mean — most reliable available)
VANILLA = {
    "BRACS_CONCH_ABMIL": 0.9270,
    "BRACS_UNI_ABMIL":   0.9243,
    "BRACS_Virchow_ABMIL": 0.9157,
    "CAM17_CONCH_ABMIL": 0.9148,
    "CAM17_UNI_ABMIL":   0.8693,
    "CAM17_Virchow_ABMIL": 0.8588,
}

def parse_config(folder_name):
    """e.g. BRACS_CONCH_ABMIL_LSAP_s1_20260521_101214_s1 → (cell=BRACS_CONCH_ABMIL, seed=1)"""
    m = re.match(r"(\w+_\w+_\w+)_LSAP_s(\d+)_\d{8}_\d{4,6}_s\d+$", folder_name)
    if m: return m.group(1), int(m.group(2))
    return None, None

per_cell_seed = defaultdict(lambda: defaultdict(dict))  # cell -> seed -> {folds: [test_auc_valloss], valauc: [test_auc_valauc]}

for d in glob.glob("/workspace/results/*_LSAP_*"):
    if not os.path.isdir(d): continue
    cell, seed = parse_config(os.path.basename(d))
    if cell is None: continue
    # collect all split_*_results.pkl
    for pkl_f in sorted(glob.glob(f"{d}/split_*_results.pkl")):
        m = re.search(r"split_(\d+)_results\.pkl", pkl_f)
        if not m: continue
        fold = int(m.group(1))
        try:
            with open(pkl_f, "rb") as fh:
                data = pickle.load(fh)
        except Exception:
            continue
        # results format varies; look for test_auc + es ablation
        # typically data is dict with per-slide labels + probs OR test_auc field
        # Easiest: check summary.csv at config root which has 'fold,test_auc' columns
    # Read summary.csv for this config
    summ = f"{d}/summary.csv"
    if os.path.exists(summ):
        try:
            with open(summ) as fh:
                lines = fh.readlines()
            if len(lines) < 2: continue
            hdr = lines[0].strip().split(",")
            test_auc_idx = hdr.index("test_auc") if "test_auc" in hdr else None
            if test_auc_idx is None: continue
            test_aucs = []
            for line in lines[1:]:
                vals = line.strip().split(",")
                if len(vals) > test_auc_idx:
                    try: test_aucs.append(float(vals[test_auc_idx]))
                    except: pass
            per_cell_seed[cell][seed]['summary_aucs'] = test_aucs
            per_cell_seed[cell][seed]['n_folds_done'] = len(test_aucs)
        except Exception:
            continue

# Aggregate per cell
print(f"\n## LSAP results harvest")
print(f"\n| Cell | n_seeds_complete | n_folds_done | LSAP AUC (5-fold mean over seeds) | Vanilla | ΔAUC |")
print(f"|---|---:|---:|---|---:|---:|")
for cell in sorted(per_cell_seed.keys()):
    # Aggregate: mean across folds, then mean across seeds
    seed_means = []
    n_folds_total = 0
    for seed, d in per_cell_seed[cell].items():
        aucs = d.get('summary_aucs', [])
        if len(aucs) > 0:
            seed_means.append(np.mean(aucs))
            n_folds_total += len(aucs)
    if not seed_means: continue
    lsap_auc_mean = np.mean(seed_means)
    lsap_auc_std = np.std(seed_means) if len(seed_means) > 1 else 0.0
    n_seeds_complete = sum(1 for s, d in per_cell_seed[cell].items() if d.get('n_folds_done', 0) == 10)
    vanilla = VANILLA.get(cell)
    delta = (lsap_auc_mean - vanilla) if vanilla else None
    delta_str = f"{delta:+.4f}" if delta is not None else "--"
    van_str = f"{vanilla:.4f}" if vanilla else "--"
    print(f"| {cell} | {n_seeds_complete}/{len(per_cell_seed[cell])} | {n_folds_total} | {lsap_auc_mean:.4f}±{lsap_auc_std:.4f} (over {len(seed_means)} seeds) | {van_str} | {delta_str} |")

# Detailed seed breakdown
print(f"\n## Per-seed folds completed")
for cell in sorted(per_cell_seed.keys()):
    print(f"\n  {cell}:")
    for seed in sorted(per_cell_seed[cell].keys()):
        d = per_cell_seed[cell][seed]
        n = d.get('n_folds_done', 0)
        aucs = d.get('summary_aucs', [])
        auc_str = f"{np.mean(aucs):.4f}" if aucs else "n/a"
        print(f"    seed={seed}: {n}/10 folds done, mean AUC = {auc_str}")
