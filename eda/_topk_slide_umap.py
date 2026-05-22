"""For each CAM16 backbone, color top-10 macro-metastasis slides differently on the UMAP
to visually judge whether their positive patches form a common manifold or per-slide blobs.

Inputs:
  /workspace/_eda_lowdim_out/cache_cam16_<bb>_gt.npz  (z_umap, slide_ids, scores, labels, ...)

Outputs:
  /workspace/_eda_lowdim_out/top10_slides_<bb>.png        (single backbone, 3 panels)
  /workspace/_eda_lowdim_out/grid_top10_slides.png        (combined 5-backbone grid)
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUT_DIR = Path("/workspace/_eda_lowdim_out")
BACKBONES = ["resnet", "ctrans", "conch", "uni", "virchow"]
SCORE_THR = 0.5
TOP_N_SLIDES = 10


def get_top10_slides(scores, slide_ids):
    mask = scores > SCORE_THR
    counts = Counter(slide_ids[mask].tolist())
    return [sid for sid, _ in counts.most_common(TOP_N_SLIDES)]


def plot_per_bb(backbone, ax_grey, ax_pos, ax_top10):
    cache = OUT_DIR / f"cache_cam16_{backbone}_gt.npz"
    if not cache.exists():
        for ax in (ax_grey, ax_pos, ax_top10):
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        return
    d = np.load(cache, allow_pickle=True)
    z = d["z_umap"]
    scores = d["scores"]
    slide_ids = d["slide_ids"]

    pos_mask = scores > SCORE_THR

    # Panel 1: all grey + positives in dark orange
    ax_grey.scatter(z[~pos_mask, 0], z[~pos_mask, 1], s=2, c="lightgrey", alpha=0.25, linewidths=0)
    ax_grey.scatter(z[pos_mask, 0], z[pos_mask, 1], s=6, c="#d95f02", alpha=0.85, linewidths=0,
                    label=f"pos>{SCORE_THR} (n={int(pos_mask.sum())})")
    ax_grey.set_title(f"{backbone}  positives (overlap > {SCORE_THR})", fontsize=9)
    ax_grey.legend(fontsize=7, framealpha=0.6, loc="best")
    ax_grey.set_xticks([]); ax_grey.set_yticks([])

    # Panel 2: positives colored by raw overlap (viridis)
    order = np.argsort(scores[pos_mask])
    pos_idx = np.where(pos_mask)[0][order]
    ax_pos.scatter(z[~pos_mask, 0], z[~pos_mask, 1], s=2, c="lightgrey", alpha=0.25, linewidths=0)
    sc = ax_pos.scatter(z[pos_idx, 0], z[pos_idx, 1], s=8, c=scores[pos_idx],
                       cmap="viridis", vmin=SCORE_THR, vmax=1.0, alpha=0.9, linewidths=0)
    ax_pos.set_title(f"{backbone}  positives by score", fontsize=9)
    ax_pos.set_xticks([]); ax_pos.set_yticks([])
    return sc, ax_top10, z, scores, slide_ids, pos_mask


def plot_top10_colors(ax_top10, z, scores, slide_ids, pos_mask, backbone):
    top10 = get_top10_slides(scores, slide_ids)
    cmap = plt.get_cmap("tab10")
    # background
    ax_top10.scatter(z[~pos_mask, 0], z[~pos_mask, 1], s=2, c="lightgrey", alpha=0.25, linewidths=0)
    # other positives in light grey
    other_pos = pos_mask & ~np.isin(slide_ids, top10)
    ax_top10.scatter(z[other_pos, 0], z[other_pos, 1], s=4, c="#777777", alpha=0.4, linewidths=0)
    # top10 each in own color
    for i, sid in enumerate(top10):
        m = pos_mask & (slide_ids == sid)
        if not m.any():
            continue
        ax_top10.scatter(z[m, 0], z[m, 1], s=14, c=[cmap(i)], alpha=0.85, linewidths=0.2,
                         edgecolors="white", label=f"{sid} (n={int(m.sum())})")
    ax_top10.set_title(f"{backbone}  top-10 macro slides", fontsize=9)
    ax_top10.legend(fontsize=6, framealpha=0.6, loc="best", ncol=2, markerscale=0.8)
    ax_top10.set_xticks([]); ax_top10.set_yticks([])


def main():
    # Combined 5x3 grid
    fig, axes = plt.subplots(len(BACKBONES), 3, figsize=(15, 4.2 * len(BACKBONES)),
                              gridspec_kw={"hspace": 0.25, "wspace": 0.06})
    for row, bb in enumerate(BACKBONES):
        cache = OUT_DIR / f"cache_cam16_{bb}_gt.npz"
        if not cache.exists():
            for ax in axes[row]:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            continue
        d = np.load(cache, allow_pickle=True)
        z = d["z_umap"]; scores = d["scores"]; slide_ids = d["slide_ids"]
        pos_mask = scores > SCORE_THR

        # Panel 1
        axes[row, 0].scatter(z[~pos_mask, 0], z[~pos_mask, 1], s=2, c="lightgrey", alpha=0.25, linewidths=0)
        axes[row, 0].scatter(z[pos_mask, 0], z[pos_mask, 1], s=6, c="#d95f02", alpha=0.85, linewidths=0)
        axes[row, 0].set_title(f"{bb}  positives (n={int(pos_mask.sum())})", fontsize=9)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        # Panel 2: positives colored by score
        order = np.argsort(scores[pos_mask])
        pos_idx_sorted = np.where(pos_mask)[0][order]
        axes[row, 1].scatter(z[~pos_mask, 0], z[~pos_mask, 1], s=2, c="lightgrey", alpha=0.25, linewidths=0)
        axes[row, 1].scatter(z[pos_idx_sorted, 0], z[pos_idx_sorted, 1], s=8, c=scores[pos_idx_sorted],
                              cmap="viridis", vmin=SCORE_THR, vmax=1.0, alpha=0.9, linewidths=0)
        axes[row, 1].set_title(f"{bb}  positives by score", fontsize=9)
        axes[row, 1].set_xticks([]); axes[row, 1].set_yticks([])

        # Panel 3: top-10 macro slides
        plot_top10_colors(axes[row, 2], z, scores, slide_ids, pos_mask, bb)

    fig.suptitle("CAM16 | positives by score and by top-10 macro slide  (per-slide signature check)", fontsize=14)
    out = OUT_DIR / "grid_top10_slides.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
