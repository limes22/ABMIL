"""Slide-level low-dim EDA: pool each slide's patch features into one vector,
then project all slides to 2D.

Two pooling variants:
  - mean: arithmetic mean across patches (global tissue signature)
  - topk: mean of top-k patches ranked by L2 norm (robust to bag-size, highlights atypical patches)

For each (dataset, backbone, pooling) -> PCA-50 + UMAP-2D, scatter by class label.

Usage:
  python _eda_lowdim_slide.py --pairs cam16:resnet,...  --pool mean,topk --topk-frac 0.05
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_ROOT = "/workspace/features"
LABEL_CSV = {
    "cam16": "/workspace/dataset_csv/camelyon16.csv",
    "tcga": "/workspace/dataset_csv/tcga_nsclc.csv",
    "cam17": "/workspace/dataset_csv/camelyon17.csv",
    "bracs": "/workspace/dataset_csv/bracs.csv",
}
LABEL_COL = {
    "cam16": "label",
    "tcga": "label",
    "cam17": "label",
    "bracs": "group",
}
FEAT_DIR = {
    ("cam16", "resnet"): "cam16_resnet",
    ("cam16", "ctrans"): "cam16_ctrans",
    ("cam16", "conch"): "cam16_conch",
    ("cam16", "uni"): "cam16_uni",
    ("cam16", "virchow"): "cam16_virchow",
    ("tcga", "resnet"): "tcga_resnet",
    ("tcga", "ctrans"): "tcga_ctrans",
    ("tcga", "conch"): "tcga_conch",
    ("tcga", "uni"): "tcga_uni",
    ("tcga", "virchow"): "tcga_virchow",
    ("cam17", "resnet"): "cam17_resnet_features",
    ("cam17", "ctrans"): "cam17_ctrans_features",
    ("cam17", "conch"): "cam17_conch_features",
    ("cam17", "uni"): "cam17_uni_features",
    ("cam17", "virchow"): "cam17_virchow_features",
    ("bracs", "resnet"): "bracs_rn50",
    ("bracs", "ctrans"): "bracs_ctrans",
    ("bracs", "conch"): "bracs_conch",
    ("bracs", "uni"): "bracs_uni",
    ("bracs", "virchow"): "bracs_virchow",
}

OUT_DIR = Path("/workspace/_eda_lowdim_slide_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_labels(dataset: str) -> dict[str, str]:
    df = pd.read_csv(LABEL_CSV[dataset])
    col = LABEL_COL.get(dataset, "label")
    return dict(zip(df["slide_id"].astype(str), df[col].astype(str)))


def pool_slide(x: np.ndarray, method: str, topk_frac: float) -> np.ndarray:
    if method == "mean":
        return x.mean(axis=0)
    elif method == "topk":
        n = x.shape[0]
        k = max(1, int(round(n * topk_frac)))
        norms = np.linalg.norm(x, axis=1)
        idx = np.argpartition(-norms, kth=min(k - 1, n - 1))[:k]
        return x[idx].mean(axis=0)
    else:
        raise ValueError(method)


def build_slide_matrix(dataset: str, backbone: str, method: str, topk_frac: float):
    feat_dir = os.path.join(FEATURES_ROOT, FEAT_DIR[(dataset, backbone)], "pt_files")
    files = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))
    labels_map = load_labels(dataset)

    vecs = []
    labels = []
    slide_ids = []
    n_patches = []
    t0 = time.time()
    for i, f in enumerate(files):
        sid = os.path.basename(f).replace(".pt", "")
        if sid not in labels_map:
            continue
        try:
            x = torch.load(f, map_location="cpu")
        except Exception as e:
            print(f"  skip {sid}: {e}", flush=True)
            continue
        if isinstance(x, dict):
            x = x.get("features", x.get("feats", None))
            if x is None:
                continue
        x = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if x.ndim != 2 or x.shape[0] == 0:
            continue
        v = pool_slide(x.astype(np.float32), method, topk_frac)
        vecs.append(v)
        labels.append(labels_map[sid])
        slide_ids.append(sid)
        n_patches.append(int(x.shape[0]))
        if (i + 1) % 200 == 0:
            print(f"  {dataset}/{backbone}/{method}: {i+1}/{len(files)} ({time.time()-t0:.1f}s)", flush=True)

    feats = np.stack(vecs, axis=0)
    labels_arr = np.array(labels)
    slide_ids_arr = np.array(slide_ids)
    n_patches_arr = np.array(n_patches)
    print(f"[{dataset}/{backbone}/{method}] slides={feats.shape[0]}, dim={feats.shape[1]}, elapsed {time.time()-t0:.1f}s", flush=True)
    return feats, labels_arr, slide_ids_arr, n_patches_arr


def run_pca(feats: np.ndarray, n_components: int = 50):
    nc = min(n_components, feats.shape[1], feats.shape[0])
    pca = PCA(n_components=nc, random_state=0)
    z = pca.fit_transform(feats)
    return z, pca.explained_variance_ratio_


def run_umap(z: np.ndarray, seed: int = 0):
    import umap
    # n_neighbors smaller for fewer points
    n_neighbors = min(15, max(5, z.shape[0] // 20))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2,
                       random_state=seed, low_memory=True, verbose=False)
    return reducer.fit_transform(z)


def plot_scatter(ax, xy, labels, title, point_size=18.0):
    classes = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=point_size, c=[cmap(i)], alpha=0.75,
                   edgecolors="white", linewidths=0.3, label=f"{c} (n={int(m.sum())})")
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=1.0, fontsize=7, loc="best", framealpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def process(dataset, backbone, method, topk_frac):
    tag = f"{dataset}_{backbone}_{method}"
    cache_path = OUT_DIR / f"cache_{tag}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        z_pca, z_umap, labels, ev, n_patches = d["z_pca"], d["z_umap"], d["labels"], d["ev"], d["n_patches"]
        sil = float(d["sil"])
        print(f"[{tag}] cached", flush=True)
    else:
        feats, labels, _, n_patches = build_slide_matrix(dataset, backbone, method, topk_frac)
        z_pca, ev = run_pca(feats, n_components=50)
        n_sil = z_pca.shape[0]
        try:
            sil = float(silhouette_score(z_pca, labels))
        except Exception:
            sil = float("nan")
        z_umap = run_umap(z_pca, seed=0)
        np.savez_compressed(cache_path, z_pca=z_pca, z_umap=z_umap, labels=labels, ev=ev,
                            n_patches=n_patches, sil=sil)
        print(f"[{tag}] done: slides={feats.shape[0]}, EV1={ev[0]:.3f}, sil={sil:.3f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_scatter(axes[0], z_pca[:, :2], labels,
                 f"{tag}  PCA (PC1 EV={ev[0]:.3f}, PC2 EV={ev[1]:.3f}, sil={sil:.3f})")
    plot_scatter(axes[1], z_umap, labels, f"{tag}  UMAP-2D")
    fig.suptitle(f"{dataset.upper()} | {backbone} | pool={method}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}.png", dpi=130)
    plt.close(fig)

    return {
        "dataset": dataset, "backbone": backbone, "pool": method,
        "n_slides": int(labels.shape[0]),
        "ev_pc1": float(ev[0]), "ev_pc2": float(ev[1]),
        "ev_top10_sum": float(ev[:10].sum()),
        "silhouette_class_pca50": sil,
        "median_n_patches": float(np.median(n_patches)),
    }


def append_stats(rows):
    stats_path = OUT_DIR / "stats.csv"
    write_header = not stats_path.exists()
    keys = ["dataset", "backbone", "pool", "n_slides", "ev_pc1", "ev_pc2",
            "ev_top10_sum", "silhouette_class_pca50", "median_n_patches"]
    with open(stats_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pool", default="mean,topk")
    ap.add_argument("--topk-frac", type=float, default=0.05)
    args = ap.parse_args()

    pairs = []
    for p in args.pairs.split(","):
        ds, bb = p.strip().split(":")
        pairs.append((ds, bb))
    pool_methods = [m.strip() for m in args.pool.split(",")]

    rows = []
    for ds, bb in pairs:
        for method in pool_methods:
            t0 = time.time()
            try:
                r = process(ds, bb, method, args.topk_frac)
                rows.append(r)
                print(f"[{ds}/{bb}/{method}] done in {time.time()-t0:.1f}s -> {r}", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[{ds}/{bb}/{method}] FAILED: {e}", flush=True)
    if rows:
        append_stats(rows)
        print(f"Appended {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
