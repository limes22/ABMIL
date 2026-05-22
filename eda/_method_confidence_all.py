#!/usr/bin/env python3
"""Per-method × cell × class confidence — Vanilla / ECC-DI / LSAP α=1.1/1.3/1.5."""
import os, glob, pickle, re, sys
import numpy as np

RESULTS = "/workspace/results"
CELLS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "BRACS_CONCH_ABMIL","BRACS_UNI_ABMIL","BRACS_Virchow_ABMIL",
    "CAM17_CONCH_ABMIL","CAM17_UNI_ABMIL","CAM17_Virchow_ABMIL"]
STAGES = {"BRACS": ["BT","AT","MT"], "CAM17": ["neg","ITC","micro","macro"]}

def find_seeds(cell, suffix):
    pat = re.compile(rf"^{cell}_{suffix}_s(\d+)_\d{{8}}_\d{{4,6}}_s\d+$")
    return [os.path.join(RESULTS, d) for d in os.listdir(RESULTS) if pat.match(d)]

def aggregate(cell, suffix):
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    folders = find_seeds(cell, suffix)
    if not folders: return None
    cc_by = {c: [] for c in range(n_cls)}
    ic_by = {c: [] for c in range(n_cls)}
    tp_by = {c: [] for c in range(n_cls)}
    nc = [0]*n_cls; nt = [0]*n_cls
    for folder in folders:
        # only count fully completed runs
        if not os.path.exists(os.path.join(folder, "split_9_results.pkl")):
            continue
        for fold in range(10):
            pkl = os.path.join(folder, f"split_{fold}_results.pkl")
            if not os.path.exists(pkl): continue
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            for sid, rec in data.items():
                if not isinstance(rec, dict) or "label" not in rec: continue
                y = int(rec["label"])
                prob = np.asarray(rec.get("prob", [])).flatten()
                if prob.shape[0] != n_cls: continue
                yhat = int(prob.argmax())
                max_p = float(prob.max()); true_p = float(prob[y])
                nt[y] += 1; tp_by[y].append(true_p)
                if yhat == y:
                    nc[y] += 1; cc_by[y].append(max_p)
                else:
                    ic_by[y].append(max_p)
    return {"nc": nc, "nt": nt, "cc": cc_by, "ic": ic_by, "tp": tp_by}

METHODS = [
    ("Vanilla", "Vanilla"),
    ("ECC-DI",  "ECCDI5"),
    ("α=1.1",   "LSAP_a11"),
    ("α=1.3",   "LSAP_a13"),
    ("α=1.5",   "LSAP_a15"),
]

for cell in CELLS:
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    print(f"\n# {cell}")
    print(f"  {'Method':<10} {'Class':<6} {'n_corr':>7} {'acc%':>6} {'conf_c':>7} {'conf_i':>7} {'P(t|y)':>7}")
    for label, suf in METHODS:
        r = aggregate(cell, suf)
        if r is None: continue
        for c in range(n_cls):
            ncc, ntc = r["nc"][c], r["nt"][c]
            if ntc == 0: continue
            cc_m = np.mean(r["cc"][c]) if r["cc"][c] else 0
            ic_m = np.mean(r["ic"][c]) if r["ic"][c] else 0
            tp_m = np.mean(r["tp"][c]) if r["tp"][c] else 0
            acc = 100 * ncc / ntc
            print(f"  {label:<10} {STAGES[ds][c]:<6} {ncc:>3}/{ntc:<3} {acc:>6.1f} {cc_m:>7.3f} {ic_m:>7.3f} {tp_m:>7.3f}")
