#!/usr/bin/env python3
"""BRACS_CONCH MT class drop EDA — why LSAP α=1.1 mis-classifies MT?"""
import os, glob, pickle, re
import numpy as np

RESULTS = "/workspace/results"
CELL = "BRACS_CONCH_ABMIL"
CLASS_NAMES = ["BT", "AT", "MT"]

def find_dirs(suffix):
    pat = re.compile(rf"^{CELL}_{suffix}_s(\d+)_\d{{8}}_\d{{4,6}}_s\d+$")
    return sorted([os.path.join(RESULTS, d) for d in os.listdir(RESULTS) if pat.match(d)])

def collect_mt(suffix):
    """For MT slides only, return (correct, predicted_class, max_prob, prob_MT)."""
    folders = find_dirs(suffix)
    if not folders: return None
    records = []  # (true=2, pred, max_prob, prob_BT, prob_AT, prob_MT, slide_id, seed, fold)
    for folder in folders:
        seed = int(re.search(r"_s(\d+)_", os.path.basename(folder)).group(1))
        for fold in range(10):
            pkl = os.path.join(folder, f"split_{fold}_results.pkl")
            if not os.path.exists(pkl): continue
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            for sid, rec in data.items():
                if not isinstance(rec, dict) or "label" not in rec: continue
                y = int(rec["label"])
                if y != 2: continue   # MT only
                prob = np.asarray(rec.get("prob", [])).flatten()
                if prob.shape[0] != 3: continue
                yhat = int(prob.argmax())
                records.append({
                    "sid": sid, "seed": seed, "fold": fold,
                    "pred": yhat, "correct": (yhat == 2),
                    "p_BT": float(prob[0]), "p_AT": float(prob[1]), "p_MT": float(prob[2]),
                    "max_prob": float(prob.max()),
                })
    return records

print("=== BRACS_CONCH MT slides analysis (all MT predictions across seeds×folds) ===\n")
methods = ["Vanilla", "ECCDI5", "LSAP_a11", "LSAP_a13", "LSAP_a15"]
results = {}
for m in methods:
    r = collect_mt(m)
    if r is None: continue
    results[m] = r
    n_total = len(r)
    n_correct = sum(1 for x in r if x["correct"])
    # confusion: MT → ?
    to_BT = sum(1 for x in r if x["pred"] == 0)
    to_AT = sum(1 for x in r if x["pred"] == 1)
    to_MT = n_correct
    # confidence
    p_MT_all = np.array([x["p_MT"] for x in r])
    p_BT_all = np.array([x["p_BT"] for x in r])
    p_AT_all = np.array([x["p_AT"] for x in r])
    # incorrect predictions: where did they go?
    wrong = [x for x in r if not x["correct"]]
    wrong_to_BT = [x for x in wrong if x["pred"] == 0]
    wrong_to_AT = [x for x in wrong if x["pred"] == 1]
    p_BT_in_wrong_BT = np.array([x["p_BT"] for x in wrong_to_BT]) if wrong_to_BT else np.array([0])
    p_AT_in_wrong_AT = np.array([x["p_AT"] for x in wrong_to_AT]) if wrong_to_AT else np.array([0])

    print(f"## {m}  ({n_total} MT predictions, {n_correct} correct = {100*n_correct/n_total:.1f}%)")
    print(f"   confusion: MT → BT {to_BT} ({100*to_BT/n_total:.1f}%)  AT {to_AT} ({100*to_AT/n_total:.1f}%)  MT {to_MT} ({100*to_MT/n_total:.1f}%)")
    print(f"   mean P(MT|y=MT): {p_MT_all.mean():.3f}  (vs P(BT)={p_BT_all.mean():.3f}, P(AT)={p_AT_all.mean():.3f})")
    print(f"   wrong→BT: n={len(wrong_to_BT)}, mean P(BT)={p_BT_in_wrong_BT.mean():.3f}")
    print(f"   wrong→AT: n={len(wrong_to_AT)}, mean P(AT)={p_AT_in_wrong_AT.mean():.3f}")
    print()

# Method comparison: same slide, different method predictions
print("=== same-slide α comparison: when does LSAP α=1.1 disagree with α=1.5 on MT? ===\n")
if "LSAP_a11" in results and "LSAP_a15" in results:
    a11 = {(x["sid"], x["seed"], x["fold"]): x for x in results["LSAP_a11"]}
    a15 = {(x["sid"], x["seed"], x["fold"]): x for x in results["LSAP_a15"]}
    common = set(a11.keys()) & set(a15.keys())
    flips_a11_wrong_a15_right = []
    flips_a11_right_a15_wrong = []
    both_wrong = []
    for k in common:
        c11 = a11[k]["correct"]; c15 = a15[k]["correct"]
        if not c11 and c15:
            flips_a11_wrong_a15_right.append((a11[k], a15[k]))
        elif c11 and not c15:
            flips_a11_right_a15_wrong.append((a11[k], a15[k]))
        elif not c11 and not c15:
            both_wrong.append((a11[k], a15[k]))
    print(f"common MT predictions (same sid+seed+fold): {len(common)}")
    print(f"  α=1.1 wrong, α=1.5 correct: {len(flips_a11_wrong_a15_right)}  ← LSAP α=1.1 의 손실")
    print(f"  α=1.1 correct, α=1.5 wrong: {len(flips_a11_right_a15_wrong)}  ← LSAP α=1.1 의 이득")
    print(f"  both wrong: {len(both_wrong)}")
    if flips_a11_wrong_a15_right:
        # detailed look at where α=1.1 specifically loses
        wrong_to_BT_a11 = [x for x, _ in flips_a11_wrong_a15_right if x["pred"] == 0]
        wrong_to_AT_a11 = [x for x, _ in flips_a11_wrong_a15_right if x["pred"] == 1]
        print(f"\n  α=1.1 wrong direction (where it fails but α=1.5 succeeds):")
        print(f"    MT → BT: {len(wrong_to_BT_a11)} cases")
        print(f"    MT → AT: {len(wrong_to_AT_a11)} cases")
        # confidence pattern
        if wrong_to_BT_a11:
            p_BT_vals = [x["p_BT"] for x in wrong_to_BT_a11]
            print(f"    when α=1.1 says BT: mean P(BT)={np.mean(p_BT_vals):.3f} (over-confident BT)")
        if wrong_to_AT_a11:
            p_AT_vals = [x["p_AT"] for x in wrong_to_AT_a11]
            print(f"    when α=1.1 says AT: mean P(AT)={np.mean(p_AT_vals):.3f}")

# Confidence calibration on MT
print("\n=== Confidence calibration of MT predictions per α ===")
print(f"{'Method':<14} {'mean P(MT|correct)':>20} {'mean P(MT|wrong)':>18}")
for m in ["LSAP_a11", "LSAP_a13", "LSAP_a15"]:
    if m not in results: continue
    r = results[m]
    p_MT_c = np.array([x["p_MT"] for x in r if x["correct"]])
    p_MT_w = np.array([x["p_MT"] for x in r if not x["correct"]])
    print(f"{m:<14} {p_MT_c.mean():>20.3f} {p_MT_w.mean():>18.3f}")
