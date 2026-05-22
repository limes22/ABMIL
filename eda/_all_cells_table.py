#!/usr/bin/env python3
"""모든 6 cells × 5 methods 현황 (완료 + 진행중)."""
import os, glob, pickle, re, csv, sys
import numpy as np

RESULTS = "/workspace/results"
CELLS_TO_PROCESS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "BRACS_CONCH_ABMIL","BRACS_UNI_ABMIL","BRACS_Virchow_ABMIL",
    "CAM17_CONCH_ABMIL","CAM17_UNI_ABMIL","CAM17_Virchow_ABMIL"]
CLASS_NAMES = {"BRACS": ["BT","AT","MT"], "CAM17": ["neg","ITC","micro","macro"]}

def find_dirs(cell, suffix):
    pat = re.compile(rf"^{cell}_{suffix}_s(\d+)_\d{{8}}_\d{{4,6}}_s\d+$")
    return sorted([os.path.join(RESULTS, d) for d in os.listdir(RESULTS) if pat.match(d)])

def per_class_pkl(pkl, n_cls):
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    correct = np.zeros(n_cls, dtype=int); total = np.zeros(n_cls, dtype=int)
    for sid, rec in data.items():
        if not isinstance(rec, dict) or "label" not in rec: continue
        y = int(rec["label"])
        prob = np.asarray(rec.get("prob", [])).flatten()
        if prob.shape[0] != n_cls: continue
        yhat = int(prob.argmax())
        total[y] += 1
        if yhat == y: correct[y] += 1
    return correct, total

def k_from_log(pat):
    ks = []
    for log in glob.glob(f"/workspace/exp_logs/{pat}.log"):
        try:
            with open(log) as f:
                for line in f:
                    m = re.search(r"k_pool mean=([0-9.]+)", line)
                    if m: ks.append(float(m.group(1)))
        except: pass
    return np.mean(ks[-30:]) if ks else None

def aggregate(cell, suffix):
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    folders = find_dirs(cell, suffix)
    if not folders: return None
    correct = np.zeros(n_cls, dtype=int); total = np.zeros(n_cls, dtype=int)
    aucs = []; ks = []; n_done_seeds = 0
    for folder in folders:
        sc = os.path.join(folder, "summary.csv")
        if os.path.exists(sc):
            for row in csv.DictReader(open(sc)):
                try: aucs.append(float(row["test_auc"]))
                except: pass
        is_done = os.path.exists(os.path.join(folder, "split_9_results.pkl"))
        if is_done: n_done_seeds += 1
        for fold in range(10):
            pkl = os.path.join(folder, f"split_{fold}_results.pkl")
            if not os.path.exists(pkl): continue
            c, t = per_class_pkl(pkl, n_cls)
            correct += c; total += t
        base = os.path.basename(folder)
        prefix = re.sub(r"_s\d+$", "", base)
        k = k_from_log(prefix)
        if k is not None: ks.append(k)
    return {
        "n_seeds_total": len(folders),
        "n_seeds_done": n_done_seeds,
        "n_runs_aucs": len(aucs),
        "auc": (np.mean(aucs), np.std(aucs)) if aucs else (None, None),
        "correct": correct, "total": total,
        "k": np.mean(ks) if ks else None,
    }

METHODS = [
    ("Vanilla",     "Vanilla"),
    ("ECCDI5",      "ECC-DI"),
    ("LSAP_a11",    "LSAP α=1.1"),
    ("LSAP_a13",    "LSAP α=1.3"),
    ("LSAP_a15",    "LSAP α=1.5(new)"),
]

for cell in CELLS_TO_PROCESS:
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    names = CLASS_NAMES[ds]
    print(f"\n## {cell}\n")
    h = f"{'Method':<18} {'seeds':>6} {'AUC':>14} {'k':>6} {'acc':>7} | " + "  ".join(f"{n:<14}" for n in names)
    print(h); print("-" * len(h))
    for suf, lab in METHODS:
        r = aggregate(cell, suf)
        if r is None: continue
        seeds = f"{r['n_seeds_done']}/{r['n_seeds_total']}"
        if r["auc"][0]:
            auc = f"{r['auc'][0]:.4f}±{r['auc'][1]:.3f}"
        else: auc = "--"
        k = f"{r['k']:.0f}" if r["k"] else "--"
        tc, tt = sum(r["correct"]), sum(r["total"])
        acc = f"{100*tc/max(tt,1):.1f}%" if tt else "--"
        pc = "  ".join(f"{c}/{t}({100*c/max(t,1):4.1f}%)" for c, t in zip(r["correct"], r["total"]))
        print(f"{lab:<18} {seeds:>6} {auc:>14} {k:>6} {acc:>7} | {pc}")
