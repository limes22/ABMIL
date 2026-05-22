"""Combined slide-level grids from cache_*.npz produced by _eda_lowdim_slide.py."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/workspace/_eda_lowdim_slide_out")
BACKBONES = ["resnet", "ctrans", "conch", "uni", "virchow"]
DATASETS = ["cam16", "tcga", "cam17", "bracs"]
POOLS = ["mean", "topk"]


def plot_scatter(ax, xy, labels, title, point_size=14.0):
    classes = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=point_size, c=[cmap(i)], alpha=0.75,
                   edgecolors="white", linewidths=0.25, label=str(c))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def build_grid(dataset, pool, method):
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(4.0 * len(BACKBONES), 4.2))
    for j, bb in enumerate(BACKBONES):
        ax = axes[j]
        cache = OUT_DIR / f"cache_{dataset}_{bb}_{pool}.npz"
        if not cache.exists():
            ax.text(0.5, 0.5, f"missing\n{cache.name}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        d = np.load(cache, allow_pickle=True)
        labels = d["labels"]
        if method == "pca":
            xy = d["z_pca"][:, :2]
            ev = d["ev"]
            sil = float(d["sil"])
            title = f"{bb} (EV1={ev[0]:.2f}, sil={sil:.2f})"
        else:
            xy = d["z_umap"]
            title = f"{bb}"
        plot_scatter(ax, xy, labels, title)
        if j == 0:
            ax.legend(markerscale=1.0, fontsize=7, loc="best", framealpha=0.6)
    fig.suptitle(f"{dataset.upper()} | pool={pool} | {method.upper()}  (slide-level)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"grid_{dataset}_{pool}_{method}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    for ds in DATASETS:
        for pool in POOLS:
            for method in ("pca", "umap"):
                build_grid(ds, pool, method)


if __name__ == "__main__":
    main()
