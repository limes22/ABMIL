#!/usr/bin/env python3
"""ACMIL α=0.020 ablation 결과 — vs 기존 α=0.010."""
import os, glob, pickle, re, csv
import numpy as np

RESULTS = "/workspace/results"
CELLS_A020 = ["BRACS_CONCH_ACMIL", "BRACS_UNI_ACMIL", "CAM17_CONCH_ACMIL", "CAM17_UNI_ACMIL"]
CLASS_NAMES = {"BRACS": ["BT","AT","MT"], "CAM17": ["neg","ITC","micro","macro"]}

def per_class_from_pkl(pkl_path, n_classes):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    correct = np.zeros(n_classes, dtype=int)
    total = np.zeros(n_classes, dtype=int)
    for sid, rec in data.items():
        if not isinstance(rec, dict): continue
        if "label" not in rec: continue
        y = int(rec["label"])
        prob = np.asarray(rec.get("prob", [])).flatten()
        if prob.shape[0] != n_classes: continue
        yhat = int(prob.argmax())
        total[y] += 1
        if yhat == y: correct[y] += 1
    return correct, total

def find_dirs(cell, suffix):
    pat = re.compile(rf"^{cell}_{suffix}_s(\d+)_\d{{8}}_\d{{4,6}}_s\d+$")
    return sorted([(int(pat.match(d).group(1)), os.path.join(RESULTS, d))
                   for d in os.listdir(RESULTS) if pat.match(d)])

def aggregate(cell, suffix):
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    folders = find_dirs(cell, suffix)
    if not folders: return None
    correct = np.zeros(n_cls, dtype=int); total = np.zeros(n_cls, dtype=int)
    aucs = []
    for seed, folder in folders:
        sc = os.path.join(folder, "summary.csv")
        if os.path.exists(sc):
            for r in csv.DictReader(open(sc)):
                try: aucs.append(float(r["test_auc"]))
                except: pass
        for fold in range(10):
            pkl = os.path.join(folder, f"split_{fold}_results.pkl")
            if not os.path.exists(pkl): continue
            c, t = per_class_from_pkl(pkl, n_cls)
            correct += c; total += t
    return {
        "n_seeds": len(folders),
        "n_runs": len(aucs),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
        "per_class_correct": correct.tolist(),
        "per_class_total": total.tolist(),
    }

for cell in CELLS_A020:
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    print(f"\n## {cell}  ({'/'.join(CLASS_NAMES[ds])})")
    for tag, suf in [("ECCDI5 (α=0.010, 기존)", "ECCDI5"),
                     ("ECCDI5_a020 (α=0.020, NEW)", "ECCDI5_a020"),
                     ("Vanilla", "Vanilla")]:
        r = aggregate(cell, suf)
        if r is None:
            print(f"  {tag:36s}  (no data)")
            continue
        auc_s = f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}" if r['auc_mean'] else "--"
        per_cls = " | ".join(f"{n}:{c}/{t}({100*c/max(t,1):.1f}%)"
                              for n, c, t in zip(CLASS_NAMES[ds], r['per_class_correct'], r['per_class_total']))
        print(f"  {tag:36s}  AUC {auc_s} ({r['n_seeds']}sd, {r['n_runs']}runs)")
        print(f"    {per_cls}")
