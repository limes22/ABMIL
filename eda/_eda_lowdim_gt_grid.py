"""Combined GT-colored grids from cache_cam16_<bb>_gt.npz files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/workspace/_eda_lowdim_out")
BACKBONES = ["resnet", "ctrans", "conch", "uni", "virchow"]


def plot_panel(ax, xy, scores, title, point_size=3.0, cmap_name="viridis"):
    order = np.argsort(scores)
    sc = ax.scatter(xy[order, 0], xy[order, 1], s=point_size, c=scores[order],
                    cmap=cmap_name, vmin=0.0, vmax=1.0, alpha=0.65, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def plot_pos_highlight(ax, xy, scores, title, ps=2.5):
    pos = scores > 0
    ax.scatter(xy[~pos, 0], xy[~pos, 1], s=ps, c="lightgrey", alpha=0.25, linewidths=0)
    if pos.any():
        order = np.argsort(scores[pos])
        idx_pos = np.where(pos)[0][order]
        ax.scatter(xy[idx_pos, 0], xy[idx_pos, 1], s=ps*3, c=scores[idx_pos],
                   cmap="plasma", vmin=0.0, vmax=1.0, alpha=0.9, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def build_grid(method: str):
    """method in {'pca','umap'}"""
    fig, axes = plt.subplots(2, len(BACKBONES), figsize=(4.0 * len(BACKBONES), 8.4),
                              gridspec_kw={"hspace": 0.18, "wspace": 0.04})
    last_sc = None
    for j, bb in enumerate(BACKBONES):
        cache = OUT_DIR / f"cache_cam16_{bb}_gt.npz"
        if not cache.exists():
            for r in range(2):
                axes[r, j].text(0.5, 0.5, f"missing\n{cache.name}", transform=axes[r,j].transAxes,
                                ha="center", va="center", fontsize=8)
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            continue
        d = np.load(cache, allow_pickle=True)
        scores = d["scores"]
        if method == "pca":
            xy = d["z_pca"][:, :2]
            ev = d["ev"]
            title = f"{bb} (EV1={ev[0]:.2f}, pos={int((scores>0).sum())})"
        else:
            xy = d["z_umap"]
            title = f"{bb} (pos={int((scores>0).sum())})"
        last_sc = plot_panel(axes[0, j], xy, scores, title)
        plot_pos_highlight(axes[1, j], xy, scores, f"{bb} (pos>0 highlighted)")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes[0, :].tolist(), location="right", fraction=0.012, pad=0.01)
        cbar.set_label("tumor overlap")
    fig.suptitle(f"CAM16 | {method.upper()} colored by GT tumor-overlap (top: continuous, bottom: highlights)",
                 fontsize=12)
    out = OUT_DIR / f"grid_cam16_gt_{method}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    for method in ("pca", "umap"):
        build_grid(method)


if __name__ == "__main__":
    main()
