"""Re-run CAM16 patch-level EDA with per-patch tumor-overlap GT scores.

Output cache `cache_cam16_{backbone}_gt.npz` includes:
  z_pca, z_umap, slide_ids, indices, tumor_scores (float [0,1]), labels (str)

Plots show UMAP colored by continuous tumor_score (viridis).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FEAT_DIR = {
    "resnet": "/workspace/features/cam16_resnet",
    "ctrans": "/workspace/features/cam16_ctrans",
    "conch": "/workspace/features/cam16_conch",
    "uni": "/workspace/features/cam16_uni",
    "virchow": "/workspace/features/cam16_virchow",
}
LABEL_CSV = "/workspace/dataset_csv/camelyon16.csv"
GT_ROOT = "/workspace/_eda_lowdim_out/gt_scores"
OUT_DIR = Path("/workspace/_eda_lowdim_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_labels():
    df = pd.read_csv(LABEL_CSV)
    return dict(zip(df["slide_id"].astype(str), df["label"].astype(str)))


def sample_with_indices(backbone: str, per_slide: int, total_cap: int, seed: int = 0):
    feat_dir = FEAT_DIR[backbone]
    pt_dir = os.path.join(feat_dir, "pt_files")
    gt_bb_dir = os.path.join(GT_ROOT, backbone)
    files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    labels_map = load_labels()
    rng = np.random.default_rng(seed)

    matched = []
    for f in files:
        sid = os.path.basename(f).replace(".pt", "")
        if sid in labels_map:
            matched.append((sid, f, labels_map[sid]))
    n_slides = len(matched)
    eff_per_slide = min(per_slide, max(1, total_cap // n_slides))
    print(f"[{backbone}] slides={n_slides}, per_slide={eff_per_slide}", flush=True)

    feats_list = []
    labels_list = []
    slide_ids_list = []
    indices_list = []
    scores_list = []
    t0 = time.time()
    n_gt_missing = 0
    for i, (sid, fpath, lbl) in enumerate(matched):
        x = torch.load(fpath, map_location="cpu")
        x = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if x.ndim != 2 or x.shape[0] == 0:
            continue
        n = x.shape[0]
        k = min(eff_per_slide, n)
        idx = rng.choice(n, size=k, replace=False) if n > k else np.arange(n)
        # tumor scores per patch from npy
        gt_path = os.path.join(gt_bb_dir, sid + ".npy")
        if os.path.exists(gt_path):
            ts = np.load(gt_path)
            if ts.size != n:
                # mismatch: pad/truncate
                ts = np.resize(ts, n)
            sel_scores = ts[idx]
        else:
            n_gt_missing += 1
            sel_scores = np.zeros(k, dtype=np.float32)
        feats_list.append(x[idx].astype(np.float32))
        labels_list.extend([lbl] * k)
        slide_ids_list.extend([sid] * k)
        indices_list.extend(idx.tolist())
        scores_list.append(sel_scores.astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{n_slides}, elapsed {time.time()-t0:.1f}s", flush=True)
    feats = np.concatenate(feats_list, axis=0)
    labels = np.array(labels_list)
    slide_ids = np.array(slide_ids_list)
    indices = np.array(indices_list, dtype=np.int64)
    scores = np.concatenate(scores_list, axis=0)
    print(f"[{backbone}] feats={feats.shape}, scores nonzero={int((scores>0).sum())}/{scores.size}, gt_missing slides={n_gt_missing}", flush=True)
    return feats, labels, slide_ids, indices, scores


def run_pca(feats, n_components=50):
    nc = min(n_components, feats.shape[1], feats.shape[0])
    pca = PCA(n_components=nc, random_state=0)
    return pca.fit_transform(feats), pca.explained_variance_ratio_


def run_umap(z, seed=0):
    import umap
    return umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=seed, low_memory=True).fit_transform(z)


def plot_gt(z_pca, z_umap, ev, labels, scores, tag):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.get_cmap("tab10")
    classes = sorted(np.unique(labels).tolist())
    for i, c in enumerate(classes):
        m = labels == c
        axes[0].scatter(z_umap[m, 0], z_umap[m, 1], s=3, c=[cmap(i)], alpha=0.4, linewidths=0, label=str(c))
    axes[0].set_title(f"{tag}  UMAP  slide-label color")
    axes[0].legend(markerscale=3.0, fontsize=8, framealpha=0.6, loc="best")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Tumor score gradient: plot zeros (low) first, then color over by descending score
    order = np.argsort(scores)
    sc = axes[1].scatter(z_umap[order, 0], z_umap[order, 1], s=3, c=scores[order],
                         cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.7, linewidths=0)
    axes[1].set_title(f"{tag}  UMAP  tumor_score (cont. [0,1])")
    cbar = plt.colorbar(sc, ax=axes[1], fraction=0.045)
    cbar.set_label("tumor overlap")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Highlight tumor-positive patches on UMAP
    pos = scores > 0
    axes[2].scatter(z_umap[~pos, 0], z_umap[~pos, 1], s=2, c="lightgrey", alpha=0.25, linewidths=0, label=f"score=0 (n={int((~pos).sum())})")
    if pos.any():
        # color by score (only positives)
        ord2 = np.argsort(scores[pos])
        idx_pos = np.where(pos)[0][ord2]
        axes[2].scatter(z_umap[idx_pos, 0], z_umap[idx_pos, 1], s=8, c=scores[idx_pos],
                        cmap="plasma", vmin=0.0, vmax=1.0, alpha=0.9, linewidths=0,
                        label=f"score>0 (n={int(pos.sum())})")
    axes[2].set_title(f"{tag}  UMAP  positive patches highlighted")
    axes[2].legend(fontsize=8, framealpha=0.6, loc="best")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.suptitle(f"CAM16 | {tag.split('_')[-1]} | PC1 EV={ev[0]:.3f}", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"{tag}_gt.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def process(backbone: str, per_slide: int, total_cap: int):
    tag = f"cam16_{backbone}"
    cache_path = OUT_DIR / f"cache_{tag}_gt.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        z_pca, z_umap = d["z_pca"], d["z_umap"]
        labels = d["labels"]; scores = d["scores"]; ev = d["ev"]
        print(f"[{tag}] cached: n={labels.size}", flush=True)
    else:
        feats, labels, slide_ids, indices, scores = sample_with_indices(backbone, per_slide, total_cap)
        z_pca, ev = run_pca(feats, n_components=50)
        print(f"[{tag}] PCA done, EV top5 = {np.round(ev[:5], 4).tolist()}", flush=True)
        z_umap = run_umap(z_pca, seed=0)
        print(f"[{tag}] UMAP done", flush=True)
        np.savez_compressed(cache_path, z_pca=z_pca, z_umap=z_umap,
                            slide_ids=slide_ids, indices=indices,
                            scores=scores, labels=labels, ev=ev)
    out = plot_gt(z_pca, z_umap, ev, labels, scores, tag)
    print(f"[{tag}] plot -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="resnet,uni,virchow")
    ap.add_argument("--per-slide", type=int, default=200)
    ap.add_argument("--total-cap", type=int, default=40000)
    args = ap.parse_args()
    bbs = [b.strip() for b in args.backbones.split(",") if b.strip()]
    for bb in bbs:
        t0 = time.time()
        try:
            process(bb, args.per_slide, args.total_cap)
            print(f"[{bb}] done in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{bb}] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
