#!/usr/bin/env python3
"""BRACS_CONCH_ABMIL + CAM17_CONCH_ABMIL 5-way 비교:
  Vanilla / ECC-DI / LSAP α=1.5 / α=1.3 / α=1.1
"""
import os, glob, pickle, re, csv
import numpy as np

RESULTS = "/workspace/results"
CELLS = ["BRACS_CONCH_ABMIL"]
CLASS_NAMES = {"BRACS": ["BT","AT","MT"], "CAM17": ["neg","ITC","micro","macro"]}

def find_dirs(cell, suffix):
    pat = re.compile(rf"^{cell}_{suffix}_s(\d+)_\d{{8}}_\d{{4,6}}_s\d+$")
    return sorted([os.path.join(RESULTS, d) for d in os.listdir(RESULTS) if pat.match(d)])

def per_class_pkl(pkl, n_cls):
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    correct = np.zeros(n_cls, dtype=int)
    total = np.zeros(n_cls, dtype=int)
    for sid, rec in data.items():
        if not isinstance(rec, dict) or "label" not in rec: continue
        y = int(rec["label"])
        prob = np.asarray(rec.get("prob", [])).flatten()
        if prob.shape[0] != n_cls: continue
        yhat = int(prob.argmax())
        total[y] += 1
        if yhat == y: correct[y] += 1
    return correct, total

def mean_k_from_log(log_path):
    if not os.path.exists(log_path): return None
    ks = []
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"k_pool mean=([0-9.]+)", line)
                if m: ks.append(float(m.group(1)))
    except: return None
    if not ks: return None
    return np.mean(ks[-20:]) if len(ks) > 20 else np.mean(ks)

def aggregate(cell, suffix):
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    folders = find_dirs(cell, suffix)
    if not folders: return None
    correct = np.zeros(n_cls, dtype=int); total = np.zeros(n_cls, dtype=int)
    aucs = []; ks = []
    for folder in folders:
        sc = os.path.join(folder, "summary.csv")
        if os.path.exists(sc):
            for row in csv.DictReader(open(sc)):
                try: aucs.append(float(row["test_auc"]))
                except: pass
        for fold in range(10):
            pkl = os.path.join(folder, f"split_{fold}_results.pkl")
            if not os.path.exists(pkl): continue
            c, t = per_class_pkl(pkl, n_cls)
            correct += c; total += t
        # k_pool from log — folder name without final _sN is the log prefix
        base = os.path.basename(folder)
        prefix = base[:-3] if re.search(r"_s\d+$", base) else base
        for log in glob.glob(f"/workspace/exp_logs/{prefix}.log"):
            k = mean_k_from_log(log)
            if k is not None: ks.append(k)
    return {
        "n_folders": len(folders),
        "n_runs": len(aucs),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
        "correct": correct.tolist(),
        "total": total.tolist(),
        "k_mean": float(np.mean(ks)) if ks else None,
    }

METHODS = [
    ("Vanilla",     "Vanilla (softmax)"),
    ("ECCDI5",      "ECC-DI"),
    ("LSAP_a11",    "LSAP α=1.1"),
    ("LSAP_a13",    "LSAP α=1.3"),
    ("LSAP_a15",    "LSAP α=1.5 (a-bisect)"),
    ("LSAP",        "LSAP α=1.5 (entmax15)"),
]

for cell in CELLS:
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    names = CLASS_NAMES[ds]
    print(f"\n## {cell}\n")
    header = f"{'Method':<26} {'AUC':>14} {'k_pool':>10} {'acc':>7}  | "
    header += "  ".join(f"{n:<11}" for n in names)
    print(header)
    print("-" * len(header))
    for suf, label in METHODS:
        r = aggregate(cell, suf)
        if r is None or r["auc_mean"] is None: continue
        auc_s = f"{r['auc_mean']:.4f}±{r['auc_std']:.3f}"
        k_s = f"{r['k_mean']:.0f}" if r["k_mean"] else "--"
        tc = sum(r["correct"]); tt = sum(r["total"])
        acc_s = f"{100*tc/max(tt,1):.1f}%"
        per_cls = "  ".join(f"{c}/{t}({100*c/max(t,1):4.1f}%)"
                            for c, t in zip(r["correct"], r["total"]))
        print(f"{label:<26} {auc_s:>14} {k_s:>10} {acc_s:>7}  | {per_cls}")
