#!/usr/bin/env python3
"""Aggregate per-cell H_norm, k, τ from V2 AdaptiveTauV2 training logs.

For each cell, walk all *_AdaptiveTauV2_s{seed}/{fold}.csv files,
extract last-epoch (after τ stabilization) values, average over folds × seeds.
"""
import os, re, glob, csv
from collections import defaultdict
import statistics

RESULTS = "/workspace/results"

# Find all AdaptiveTauV2 result folders
folders = glob.glob(f"{RESULTS}/*_AdaptiveTauV2_s*")
print(f"found {len(folders)} V2 folders")

per_cell = defaultdict(lambda: {
    "H_norm": [], "mean_k": [], "mean_kN_ratio": [], "tau": [], "sharp_ratio": [], "n_slides": [],
})

for fold_dir in folders:
    name = os.path.basename(fold_dir)
    # Strip trailing _s{seed}_{ts}_{s{seed}} → cell = e.g. CAM17_UNI_CLAM
    m = re.match(r"(.+)_AdaptiveTauV2_s\d+_\d{8}_\d{4,6}_s\d+$", name)
    if not m: continue
    cell = m.group(1)

    # Each fold has its own alpha_traj csv: s_{fold}_alpha_traj.csv
    for csv_path in glob.glob(f"{fold_dir}/s_*_alpha_traj.csv"):
        # Read last 5 epochs and average for stability
        rows = []
        try:
            with open(csv_path) as f:
                rd = csv.DictReader(f)
                for r in rd:
                    rows.append(r)
        except Exception:
            continue
        if len(rows) < 5: continue
        last5 = rows[-5:]
        try:
            H = statistics.mean(float(r["mean_H_norm"]) for r in last5 if r.get("mean_H_norm"))
            k = statistics.mean(float(r["mean_k"]) for r in last5 if r.get("mean_k"))
            kN = statistics.mean(float(r["mean_kN_ratio"]) for r in last5 if r.get("mean_kN_ratio"))
            sr = statistics.mean(float(r["sharp_ratio"]) for r in last5 if r.get("sharp_ratio"))
            ns = int(float(last5[-1]["n_slides"])) if last5[-1].get("n_slides") else 0
        except Exception:
            continue
        per_cell[cell]["H_norm"].append(H)
        per_cell[cell]["mean_k"].append(k)
        per_cell[cell]["mean_kN_ratio"].append(kN)
        per_cell[cell]["sharp_ratio"].append(sr)
        per_cell[cell]["n_slides"].append(ns)

    # Also read final τ from adaptive_tau.csv (latest periodic_done)
    for tau_path in glob.glob(f"{fold_dir}/s_*_adaptive_tau.csv"):
        try:
            with open(tau_path) as f:
                rd = csv.DictReader(f)
                tau_vals = [float(r["tau_set"]) for r in rd if r.get("tau_set", "").strip()]
            if tau_vals:
                per_cell[cell]["tau"].append(tau_vals[-1])  # use last (most stable)
        except Exception:
            continue

# Print summary table
header_cells = ["cell", "n_data", "n_slides", "mean_H_norm", "mean_k", "k/N%", "tau_table",
                "tau_V2_last_mean", "sharp_ratio%"]
print("\t".join(header_cells))
rows_out = []
for cell, d in sorted(per_cell.items()):
    if not d["H_norm"]: continue
    n_data = len(d["H_norm"])
    H = statistics.mean(d["H_norm"])
    k = statistics.mean(d["mean_k"])
    kN = statistics.mean(d["mean_kN_ratio"])
    sr = statistics.mean(d["sharp_ratio"])
    ns = statistics.mean(d["n_slides"]) if d["n_slides"] else 0
    tau = statistics.mean(d["tau"]) if d["tau"] else None
    rows_out.append({
        "cell": cell, "n_data": n_data, "n_slides": int(ns),
        "mean_H_norm": H, "mean_k": k, "k/N%": kN*100,
        "tau_V2_last_mean": tau, "sharp_ratio%": sr*100,
    })
    tau_str = f"{tau:.3f}" if tau is not None else "--"
    print(f"{cell}\t{n_data}\t{int(ns)}\t{H:.3f}\t{k:.1f}\t{kN*100:.2f}\t--\t{tau_str}\t{sr*100:.1f}")

# Save raw
import json
with open("/workspace/_tau_table_stats.json", "w") as f:
    json.dump(rows_out, f, indent=2, default=str)
print(f"\nSaved /workspace/_tau_table_stats.json — {len(rows_out)} cells")
