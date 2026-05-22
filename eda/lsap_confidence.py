#!/usr/bin/env python3
"""LSAP per-cell × class confidence 분석 (val_loss ckpt).
Confidence = max(prob) = predicted class probability.
   - correct vs incorrect 별 confidence
   - class 별 mean confidence
"""
import os, glob, pickle, re
import numpy as np

RESULTS = "/workspace/results"
CELLS = ["BRACS_CONCH_ABMIL", "BRACS_UNI_ABMIL", "BRACS_Virchow_ABMIL",
         "CAM17_CONCH_ABMIL", "CAM17_UNI_ABMIL", "CAM17_Virchow_ABMIL"]
STAGES = {"BRACS": ["BT", "AT", "MT"], "CAM17": ["neg", "ITC", "micro", "macro"]}

def find_seeds(cell, suffix):
    pat = re.compile(rf"^{cell}_{suffix}_s(\d+)_\d{{8}}_\d{{4,6}}_s\d+$")
    return [(int(pat.match(d).group(1)), os.path.join(RESULTS, d))
            for d in os.listdir(RESULTS) if pat.match(d)]

def aggregate_confidence(cell, suffix):
    """Aggregate per-class confidence across all seeds × folds (val_loss ckpt)."""
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    folders = find_seeds(cell, suffix)
    if not folders: return None
    # class-level: predicted-confidence (max prob) for each slide
    # also: true-class-confidence (prob[true_label])
    by_class_correct = {c: [] for c in range(n_cls)}      # confidence of correctly predicted slides per class
    by_class_incorrect = {c: [] for c in range(n_cls)}    # confidence of incorrectly predicted slides per class
    true_class_prob = {c: [] for c in range(n_cls)}       # prob[true_label] regardless of prediction
    pred_prob_overall = []
    for seed, folder in folders:
        for fold in range(10):
            pkl = os.path.join(folder, f"split_{fold}_results.pkl")
            if not os.path.exists(pkl): continue
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            for sid, rec in data.items():
                if not isinstance(rec, dict): continue
                if "label" not in rec or "prob" not in rec: continue
                y = int(rec["label"])
                prob = np.asarray(rec["prob"]).flatten()
                if prob.shape[0] != n_cls: continue
                yhat = int(prob.argmax())
                max_p = float(prob.max())
                true_p = float(prob[y])
                pred_prob_overall.append(max_p)
                if yhat == y:
                    by_class_correct[y].append(max_p)
                else:
                    by_class_incorrect[y].append(max_p)
                true_class_prob[y].append(true_p)
    return {
        "by_class_correct": by_class_correct,
        "by_class_incorrect": by_class_incorrect,
        "true_class_prob": true_class_prob,
        "n_seeds": len(folders),
    }

print(f"{'Cell':<22} {'Stage':<7} {'n_corr':>6} {'conf(corr)':>11} {'n_inc':>6} {'conf(inc)':>10} | {'P(true|y)':>10}")
print("-" * 90)
for cell in CELLS:
    ds = "BRACS" if cell.startswith("BRACS") else "CAM17"
    n_cls = 3 if ds == "BRACS" else 4
    r = aggregate_confidence(cell, "LSAP")
    if r is None:
        print(f"{cell}: no LSAP folder"); continue
    print(f"# {cell} (n_seeds={r['n_seeds']})")
    for c in range(n_cls):
        name = STAGES[ds][c]
        cc = r["by_class_correct"][c]
        ic = r["by_class_incorrect"][c]
        tp = r["true_class_prob"][c]
        cc_str = f"{np.mean(cc):.3f}" if cc else "--"
        ic_str = f"{np.mean(ic):.3f}" if ic else "--"
        tp_str = f"{np.mean(tp):.3f}" if tp else "--"
        print(f"{'':<22} {name:<7} {len(cc):>6} {cc_str:>11} {len(ic):>6} {ic_str:>10} | {tp_str:>10}")
    print()
