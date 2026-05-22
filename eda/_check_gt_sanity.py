"""Sanity checks for CAM16 GT-based EDA:
  (1) Per-slide tumor-patch ratio distribution
  (2) Yellow-cluster slide concentration (which slides own the high-overlap patches?)
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

GT_ROOT = Path("/workspace/_eda_lowdim_out/gt_scores")
OUT_DIR = Path("/workspace/_eda_lowdim_out")


def per_slide_ratios(backbone: str):
    """Print per-slide positive ratio distribution for all tumor slides + summary."""
    bb_dir = GT_ROOT / backbone
    files = sorted(f for f in os.listdir(bb_dir) if f.endswith(".npy"))
    rows = []
    for fn in files:
        sid = fn[:-4]
        scores = np.load(bb_dir / fn)
        n = scores.size
        n_pos = int((scores > 0).sum())
        n_strict = int((scores >= 0.5).sum())  # >=50% overlap
        ratio = n_pos / max(n, 1)
        rows.append((sid, n, n_pos, n_strict, ratio, float(scores.max()), float(scores[scores > 0].mean()) if n_pos else 0.0))
    rows_tumor = [r for r in rows if r[0].startswith("tumor")]
    rows_normal = [r for r in rows if r[0].startswith("normal")]
    rows_test = [r for r in rows if r[0].startswith("test")]

    def summarize(name, rs):
        if not rs: return
        ratios = np.array([r[4] for r in rs])
        n_with_pos = int((ratios > 0).sum())
        print(f"  {name}: {len(rs)} slides, {n_with_pos} have any positive patch")
        if n_with_pos:
            r_pos = ratios[ratios > 0]
            qs = np.quantile(r_pos, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
            print(f"    positive-ratio (only slides with pos>0): min={qs[0]:.4f}, p10={qs[1]:.4f}, p25={qs[2]:.4f}, "
                  f"median={qs[3]:.4f}, p75={qs[4]:.4f}, p90={qs[5]:.4f}, max={qs[6]:.4f}")
        # tail (top-10 by positive ratio)
        rs_sorted = sorted(rs, key=lambda r: -r[4])
        print(f"    top-10 by ratio:")
        for sid, n, np_, ns, r, mx, mean_pos in rs_sorted[:10]:
            print(f"      {sid}: pos={np_}/{n} ({r:.2%}), strict(>=0.5)={ns}, max_overlap={mx:.2f}, mean(>0)={mean_pos:.2f}")

    print(f"\n=== {backbone} ===")
    summarize("tumor_*", rows_tumor)
    summarize("test_*", rows_test)
    summarize("normal_* (should be all 0)", rows_normal)


def cluster_concentration(backbone: str, score_thresh: float = 0.5):
    """In the cached 40k samples, how many distinct slides contribute to score>thresh patches?"""
    cache = OUT_DIR / f"cache_cam16_{backbone}_gt.npz"
    if not cache.exists():
        print(f"  cache missing: {cache}")
        return
    d = np.load(cache, allow_pickle=True)
    scores = d["scores"]
    slide_ids = d["slide_ids"]
    mask = scores > score_thresh
    n_yellow = int(mask.sum())
    if n_yellow == 0:
        print(f"  no patches with score > {score_thresh}")
        return
    contributing = slide_ids[mask]
    counts = Counter(contributing.tolist())
    top = counts.most_common()
    cum = 0
    n_slides = len(counts)
    print(f"\n--- {backbone}: {n_yellow} sampled patches with score > {score_thresh}, "
          f"from {n_slides} distinct slides ---")
    print(f"  top-10 contributors:")
    for sid, c in top[:10]:
        cum += c
        print(f"    {sid}: {c} ({100*c/n_yellow:.1f}%, cum={100*cum/n_yellow:.1f}%)")
    # 50%, 90% concentration
    cum = 0
    n50 = n90 = None
    for i, (_, c) in enumerate(top):
        cum += c
        if n50 is None and cum >= 0.5 * n_yellow:
            n50 = i + 1
        if n90 is None and cum >= 0.9 * n_yellow:
            n90 = i + 1
            break
    print(f"  {n50} slide(s) own 50% of yellow patches; {n90} slides own 90%")


def main():
    bbs = ["resnet", "ctrans", "conch", "uni", "virchow"]
    for bb in bbs:
        per_slide_ratios(bb)
    print("\n\n=== CLUSTER CONCENTRATION (score>0.5) ===")
    for bb in bbs:
        cluster_concentration(bb, score_thresh=0.5)
    print("\n\n=== CLUSTER CONCENTRATION (score>0) ===")
    for bb in bbs:
        cluster_concentration(bb, score_thresh=0.0)


if __name__ == "__main__":
    main()
