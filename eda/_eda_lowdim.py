"""EDA: low-dim projection of patch features per (dataset, backbone).

Per (dataset, backbone) combo:
  1. Load slide labels from dataset_csv/{dataset}.csv
  2. Sample N patches per slide (cap total to ~TOTAL_CAP)
  3. PCA -> 50d (record EV); UMAP 2D on PCA-50
  4. Scatter colored by class label
  5. Cache reduced arrays to .npz
  6. Append summary stats (EV, silhouette in PCA-50, norm stats) to stats.csv

Usage:
  python _eda_lowdim.py --pairs cam16:resnet,cam16:uni,...
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
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
# Map (dataset, backbone) -> feature directory name (under FEATURES_ROOT)
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

OUT_DIR = Path("/workspace/_eda_lowdim_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_labels(dataset: str) -> dict[str, str]:
    df = pd.read_csv(LABEL_CSV[dataset])
    col = LABEL_COL.get(dataset, "label")
    return dict(zip(df["slide_id"].astype(str), df[col].astype(str)))


def sample_features(dataset: str, backbone: str, per_slide: int, total_cap: int, seed: int = 0):
    feat_dir = os.path.join(FEATURES_ROOT, FEAT_DIR[(dataset, backbone)], "pt_files")
    files = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))
    labels_map = load_labels(dataset)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    matched = []
    for f in files:
        sid = os.path.basename(f).replace(".pt", "")
        if sid in labels_map:
            matched.append((sid, f, labels_map[sid]))
    n_slides = len(matched)
    if n_slides == 0:
        raise RuntimeError(f"No matched slides for {dataset}/{backbone} in {feat_dir}")

    # Compute per-slide cap so that expected total ~ total_cap (still respect user per_slide upper bound)
    eff_per_slide = min(per_slide, max(1, total_cap // n_slides))
    print(f"[{dataset}/{backbone}] slides={n_slides}, per_slide={eff_per_slide} (cap={per_slide}, total_cap={total_cap})", flush=True)

    feats_list = []
    labels_list = []
    slide_ids_list = []
    t0 = time.time()
    for i, (sid, fpath, lbl) in enumerate(matched):
        try:
            x = torch.load(fpath, map_location="cpu")
        except Exception as e:
            print(f"  skip {sid}: {e}", flush=True)
            continue
        if isinstance(x, dict):
            x = x.get("features", x.get("feats", None))
            if x is None:
                continue
        x = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if x.ndim != 2:
            continue
        n = x.shape[0]
        k = min(eff_per_slide, n)
        idx = np_rng.choice(n, size=k, replace=False) if n > k else np.arange(n)
        feats_list.append(x[idx].astype(np.float32))
        labels_list.extend([lbl] * k)
        slide_ids_list.extend([sid] * k)
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{n_slides} slides, elapsed {time.time()-t0:.1f}s", flush=True)

    feats = np.concatenate(feats_list, axis=0)
    labels_arr = np.array(labels_list)
    slide_ids_arr = np.array(slide_ids_list)
    print(f"[{dataset}/{backbone}] sampled feats shape={feats.shape}, elapsed {time.time()-t0:.1f}s", flush=True)
    return feats, labels_arr, slide_ids_arr


def run_pca(feats: np.ndarray, n_components: int = 50):
    n = feats.shape[0]
    nc = min(n_components, feats.shape[1], n)
    pca = PCA(n_components=nc, random_state=0)
    z = pca.fit_transform(feats)
    return z, pca.explained_variance_ratio_


def run_umap(z_pca: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1, seed: int = 0):
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed,
                       low_memory=True, verbose=False)
    return reducer.fit_transform(z_pca)


def plot_scatter(ax, xy: np.ndarray, labels: np.ndarray, title: str, point_size: float = 3.0):
    classes = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=point_size, c=[cmap(i)], alpha=0.45, label=f"{c} (n={int(m.sum())})", linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=2.5, fontsize=7, loc="best", framealpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def process_pair(dataset: str, backbone: str, per_slide: int, total_cap: int):
    tag = f"{dataset}_{backbone}"
    cache_path = OUT_DIR / f"cache_{tag}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        z_pca = d["z_pca"]
        z_umap = d["z_umap"]
        labels = d["labels"]
        ev = d["ev"]
        norms = d["norms"]
        sil = float(d["sil"])
        print(f"[{tag}] loaded cache: pca={z_pca.shape}, umap={z_umap.shape}", flush=True)
    else:
        feats, labels, _ = sample_features(dataset, backbone, per_slide, total_cap)
        norms = np.linalg.norm(feats, axis=1)
        z_pca, ev = run_pca(feats, n_components=50)
        print(f"[{tag}] PCA done, EV top5 = {np.round(ev[:5], 4).tolist()}", flush=True)
        # silhouette on a subsample for cost
        n_sil = min(5000, z_pca.shape[0])
        idx_sil = np.random.default_rng(0).choice(z_pca.shape[0], size=n_sil, replace=False)
        try:
            sil = float(silhouette_score(z_pca[idx_sil], labels[idx_sil]))
        except Exception:
            sil = float("nan")
        z_umap = run_umap(z_pca, n_neighbors=30, min_dist=0.1, seed=0)
        print(f"[{tag}] UMAP done", flush=True)
        np.savez_compressed(cache_path, z_pca=z_pca, z_umap=z_umap, labels=labels, ev=ev, norms=norms, sil=sil)

    # individual figure: PCA + UMAP side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_scatter(axes[0], z_pca[:, :2], labels, f"{tag}  PCA (PC1 EV={ev[0]:.3f}, PC2 EV={ev[1]:.3f})")
    plot_scatter(axes[1], z_umap, labels, f"{tag}  UMAP-2D")
    fig.suptitle(f"{dataset.upper()} | {backbone}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}.png", dpi=130)
    plt.close(fig)

    return {
        "dataset": dataset,
        "backbone": backbone,
        "n_samples": int(labels.shape[0]),
        "ev_pc1": float(ev[0]),
        "ev_pc2": float(ev[1]),
        "ev_top10_sum": float(ev[:10].sum()),
        "silhouette_class_pca50": sil,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_median": float(np.median(norms)),
    }


def append_stats(rows: list[dict]):
    stats_path = OUT_DIR / "stats.csv"
    write_header = not stats_path.exists()
    keys = ["dataset", "backbone", "n_samples", "ev_pc1", "ev_pc2", "ev_top10_sum",
            "silhouette_class_pca50", "norm_mean", "norm_std", "norm_median"]
    with open(stats_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="comma-separated dataset:backbone, e.g. cam16:resnet,tcga:uni")
    ap.add_argument("--per-slide", type=int, default=200)
    ap.add_argument("--total-cap", type=int, default=40000)
    args = ap.parse_args()

    pairs = []
    for p in args.pairs.split(","):
        ds, bb = p.strip().split(":")
        pairs.append((ds, bb))

    print(f"Pairs: {pairs}", flush=True)
    rows = []
    for ds, bb in pairs:
        t0 = time.time()
        try:
            r = process_pair(ds, bb, args.per_slide, args.total_cap)
            rows.append(r)
            print(f"[{ds}/{bb}] done in {time.time()-t0:.1f}s -> {r}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{ds}/{bb}] FAILED: {e}", flush=True)
    if rows:
        append_stats(rows)
        print(f"Appended {len(rows)} rows to {OUT_DIR / 'stats.csv'}", flush=True)


if __name__ == "__main__":
    main()
