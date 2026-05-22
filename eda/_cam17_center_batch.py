"""CAM17 5-center batch analysis (label-based, no GT needed).

For each backbone:
  Pool = sample N patches per negative slide (label='negative') across all 5 centers.
  k-NN in RAW feature space.
  Classify each neighbor by (same_slide?, same_center?):
    SS  = same slide
    SCDS = same center, different slide
    DC  = different center
  Aggregate per-backbone and per-center.

  Headlines:
    - mean SS = within-slide tightness (slide continuity)
    - mean (SCDS - random) = center batch above slide-binding
    - per-center variation in SS = which centers are more slide-bound

  Secondary (exploratory):
    - Repeat the kNN bucket on TUMOR slides (label in itc/micro/macro), sampling random patches.
      These are mostly normal tissue with small tumor inclusions. cs_kNN here vs negative
      tells whether slide-binding is stronger on tumor slides (consistent with CAM16 step B).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

LABEL_CSV = "/workspace/dataset_csv/camelyon17.csv"
STAGES_CSV = "/workspace/data/CAMELYON17/stages.csv"
FEAT_PT_DIR = {
    "resnet": "/workspace/features/cam17_resnet_features/pt_files",
    "ctrans": "/workspace/features/cam17_ctrans_features/pt_files",
    "conch": "/workspace/features/cam17_conch_features/pt_files",
    "uni": "/workspace/features/cam17_uni_features/pt_files",
    "virchow": "/workspace/features/cam17_virchow_features/pt_files",
}
OUT_DIR = Path("/workspace/_eda_lowdim_out/cam17_center")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_SLIDE = 100
K_NN = 11


def load_meta():
    # slide_id -> label (4-class)
    df = pd.read_csv(LABEL_CSV)
    label_by_slide = dict(zip(df["slide_id"].astype(str), df["label"].astype(str)))
    # slide_id -> center
    center_by_slide = {}
    with open(STAGES_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["patient"]
            if not sid.endswith(".tif"):
                continue
            sid = sid.replace(".tif", "")
            center_by_slide[sid] = int(row["center"])
    return label_by_slide, center_by_slide


def build_pool(backbone: str, label_by_slide, center_by_slide, select_label="negative", seed=0):
    feat_dir = FEAT_PT_DIR[backbone]
    files = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))
    rng = np.random.default_rng(seed)
    X = []; slides = []; centers = []
    n_skip_no_label = 0; n_skip_no_center = 0; n_kept = 0
    for f in files:
        sid = os.path.basename(f).replace(".pt", "")
        lbl = label_by_slide.get(sid)
        if lbl is None:
            n_skip_no_label += 1; continue
        if select_label == "negative" and lbl != "negative":
            continue
        if select_label == "any_tumor" and lbl == "negative":
            continue
        if select_label not in ("negative", "any_tumor"):
            if lbl != select_label:
                continue
        if sid not in center_by_slide:
            n_skip_no_center += 1; continue
        try:
            x = torch.load(f, map_location="cpu")
        except Exception as e:
            print(f"  ! {sid}: {e}"); continue
        x = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if x.ndim != 2 or x.shape[0] == 0:
            continue
        k = min(PER_SLIDE, x.shape[0])
        idx = rng.choice(x.shape[0], k, replace=False)
        X.append(x[idx].astype(np.float32))
        slides.extend([sid] * k)
        centers.extend([center_by_slide[sid]] * k)
        n_kept += 1
    if not X:
        return None, None, None, {"n_kept": 0}
    X = np.concatenate(X, axis=0)
    slides = np.array(slides)
    centers = np.array(centers, dtype=np.int64)
    meta = {
        "n_slides_kept": n_kept,
        "n_patches": int(X.shape[0]),
        "n_skip_no_label": n_skip_no_label,
        "n_skip_no_center": n_skip_no_center,
        "center_slide_counts": {int(c): int(((centers == c) & np.concatenate([[True], np.diff(slides) != ""])).sum())  # placeholder
                                  for c in np.unique(centers)},
        "patches_per_center": {int(c): int((centers == c).sum()) for c in np.unique(centers)},
    }
    return X, slides, centers, meta


def kNN_buckets(X, slides, centers, k=K_NN):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    _, nbr_idx = nn.kneighbors(X)
    buckets = np.zeros((X.shape[0], 3), dtype=np.float32)  # SS, SCDS, DC
    for i in range(X.shape[0]):
        for j in nbr_idx[i, 1:]:
            if slides[j] == slides[i]:
                buckets[i, 0] += 1
            elif centers[j] == centers[i]:
                buckets[i, 1] += 1
            else:
                buckets[i, 2] += 1
    buckets /= (k - 1)
    return buckets


def analytic_baseline(slides, centers):
    """Expected SS, SCDS, DC under random slide assignment (uniform over pool)."""
    n = len(slides)
    counts_s = Counter(slides.tolist())
    counts_c = Counter(centers.tolist())
    # P(same_slide) = sum n_s * (n_s - 1) / (n * (n-1))
    p_ss = sum(c * (c - 1) for c in counts_s.values()) / (n * (n - 1))
    # P(same_center) similar
    p_sc = sum(c * (c - 1) for c in counts_c.values()) / (n * (n - 1))
    # SCDS = same center but not same slide
    p_scds = p_sc - p_ss
    p_dc = 1 - p_sc
    return {"SS": p_ss, "SCDS": p_scds, "DC": p_dc}


def process(backbone: str, label_by_slide, center_by_slide):
    print(f"\n========= {backbone} =========")
    res = {}

    for label_grp in ["negative", "any_tumor"]:
        print(f"--- {backbone}/{label_grp} ---")
        t0 = time.time()
        X, slides, centers, meta = build_pool(backbone, label_by_slide, center_by_slide, select_label=label_grp)
        if X is None:
            print(f"  no pool, skip"); continue
        print(f"  n_patches={X.shape[0]} from {meta['n_slides_kept']} slides; "
              f"per_center={meta['patches_per_center']}; load_t={time.time()-t0:.1f}s", flush=True)

        buckets = kNN_buckets(X, slides, centers)
        m = buckets.mean(0)
        baseline = analytic_baseline(slides, centers)
        print(f"  mean: SS={m[0]:.3f}  SCDS={m[1]:.3f}  DC={m[2]:.3f}  (cs_kNN = SCDS+DC = {m[1]+m[2]:.3f})")
        print(f"  random baseline: SS={baseline['SS']:.4f}  SCDS={baseline['SCDS']:.4f}  DC={baseline['DC']:.4f}")
        print(f"  enrichment SS / baseline = {m[0]/max(baseline['SS'],1e-9):.1f}x")
        if baseline['SCDS'] > 0:
            print(f"  enrichment SCDS / baseline = {m[1]/max(baseline['SCDS'],1e-9):.2f}x  (>1 => center batch above slide)")

        # Per-center
        per_center = {}
        for c in sorted(np.unique(centers).tolist()):
            mk = centers == c
            b = buckets[mk].mean(0)
            per_center[int(c)] = {
                "n_patches": int(mk.sum()),
                "n_slides": int(len(np.unique(slides[mk]))),
                "SS": float(b[0]), "SCDS": float(b[1]), "DC": float(b[2]),
                "cs_kNN": float(b[1] + b[2]),
            }
            print(f"    center {c}: n_pat={per_center[int(c)]['n_patches']:5d}, n_sld={per_center[int(c)]['n_slides']:3d}, "
                  f"SS={b[0]:.3f}, SCDS={b[1]:.3f}, DC={b[2]:.3f}, cs_kNN={b[1]+b[2]:.3f}")

        res[label_grp] = {
            "n_patches": int(X.shape[0]),
            "n_slides": meta["n_slides_kept"],
            "mean": {"SS": float(m[0]), "SCDS": float(m[1]), "DC": float(m[2]), "cs_kNN": float(m[1]+m[2])},
            "baseline": baseline,
            "enrichment_SS": float(m[0]/max(baseline['SS'],1e-9)),
            "enrichment_SCDS": float(m[1]/max(baseline['SCDS'],1e-9)),
            "per_center": per_center,
        }

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="resnet,ctrans,conch,uni,virchow")
    args = ap.parse_args()
    label_by_slide, center_by_slide = load_meta()
    print(f"loaded {len(label_by_slide)} slide labels, {len(center_by_slide)} center mappings", flush=True)
    print(f"label dist: {Counter(label_by_slide.values())}", flush=True)

    bbs = [b.strip() for b in args.backbones.split(",") if b.strip()]
    all_res = {}
    for bb in bbs:
        all_res[bb] = process(bb, label_by_slide, center_by_slide)

    out = OUT_DIR / "center_batch_summary.json"
    with open(out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {out}")

    # Summary table
    print("\n=== CAM17 negative-slide cs_kNN summary (5-center pool) ===")
    print(f"{'backbone':<10} {'mean_SS':>8} {'mean_SCDS':>10} {'mean_DC':>8} {'cs_kNN':>8} {'enrich_SS':>10} {'enrich_SCDS':>12}")
    for bb in bbs:
        r = all_res.get(bb, {}).get("negative")
        if not r: continue
        m = r["mean"]
        print(f"{bb:<10} {m['SS']:>8.3f} {m['SCDS']:>10.3f} {m['DC']:>8.3f} {m['cs_kNN']:>8.3f} "
              f"{r['enrichment_SS']:>10.1f} {r['enrichment_SCDS']:>12.2f}")
    print("\n=== CAM17 tumor-slide cs_kNN (label in itc/micro/macro) ===")
    print(f"{'backbone':<10} {'mean_SS':>8} {'mean_SCDS':>10} {'mean_DC':>8} {'cs_kNN':>8}")
    for bb in bbs:
        r = all_res.get(bb, {}).get("any_tumor")
        if not r: continue
        m = r["mean"]
        print(f"{bb:<10} {m['SS']:>8.3f} {m['SCDS']:>10.3f} {m['DC']:>8.3f} {m['cs_kNN']:>8.3f}")


if __name__ == "__main__":
    main()
