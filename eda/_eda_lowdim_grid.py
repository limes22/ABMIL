"""Build combined grid figures from cache_*.npz files written by _eda_lowdim.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/workspace/_eda_lowdim_out")
BACKBONES = ["resnet", "ctrans", "conch", "uni", "virchow"]
DATASETS = ["cam16", "tcga", "cam17", "bracs"]


def plot_scatter(ax, xy, labels, title, point_size=2.0):
    classes = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=point_size, c=[cmap(i)], alpha=0.4, linewidths=0, label=str(c))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if title.endswith("(legend)"):
        pass


def build_grid(dataset: str, method: str):
    # method in {"pca", "umap"}
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(4.0 * len(BACKBONES), 4.2))
    found_any = False
    for j, bb in enumerate(BACKBONES):
        ax = axes[j]
        cache = OUT_DIR / f"cache_{dataset}_{bb}.npz"
        if not cache.exists():
            ax.text(0.5, 0.5, f"missing\n{cache.name}", transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        d = np.load(cache, allow_pickle=True)
        labels = d["labels"]
        if method == "pca":
            xy = d["z_pca"][:, :2]
            ev = d["ev"]
            title = f"{bb} (PC1 EV={ev[0]:.2f}, PC2 EV={ev[1]:.2f})"
        else:
            xy = d["z_umap"]
            title = f"{bb}"
        plot_scatter(ax, xy, labels, title)
        if j == 0:
            ax.legend(markerscale=3.0, fontsize=7, loc="best", framealpha=0.6)
        found_any = True
    fig.suptitle(f"{dataset.upper()} | {method.upper()} across backbones", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"grid_{dataset}_{method}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out} (found_any={found_any})", flush=True)


def main():
    for ds in DATASETS:
        for method in ("pca", "umap"):
            build_grid(ds, method)


if __name__ == "__main__":
    main()
