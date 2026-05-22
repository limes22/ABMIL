"""Group per-slide ST by CAM16 center (RUMC=0, UMCU=1)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_CSV = "/tmp/cam16_ref.csv"
PSDIR = Path("/workspace/_eda_lowdim_out/manifold/per_slide")
BBS = ["resnet", "ctrans", "conch", "uni", "virchow"]


def main():
    center_by_slide = {}
    with open(REF_CSV) as f:
        for row in csv.DictReader(f):
            center_by_slide[row["image"].replace(".tif", "")] = int(row["center"])

    all_data = {}  # bb -> {sid: (ST, OT, n, center)}
    summary_rows = []
    for bb in BBS:
        p = PSDIR / f"{bb}_per_slide.csv"
        if not p.exists():
            print(f"missing {p}"); continue
        d = {}
        with open(p) as f:
            for row in csv.DictReader(f):
                sid = row["slide_id"]
                c = center_by_slide.get(sid, -1)
                if c < 0:
                    continue
                d[sid] = {
                    "ST": float(row["ST"]),
                    "OT": float(row["OT"]),
                    "n_tumor_patches": int(row["n_tumor_patches"]),
                    "center": c,
                }
        all_data[bb] = d
        # per-center stats (weight equally by slide)
        for c in (0, 1):
            sts = [v["ST"] for v in d.values() if v["center"] == c]
            ots = [v["OT"] for v in d.values() if v["center"] == c]
            if not sts: continue
            summary_rows.append({
                "backbone": bb, "center": c, "n_slides": len(sts),
                "ST_mean": np.mean(sts), "ST_median": np.median(sts),
                "OT_mean": np.mean(ots), "OT_median": np.median(ots),
            })

    # Print table
    print(f"{'backbone':<10} {'center':>7} {'n':>4} {'ST mean':>9} {'ST med':>8} {'OT mean':>9} {'OT med':>8}")
    for r in summary_rows:
        print(f"{r['backbone']:<10} {r['center']:>7} {r['n_slides']:>4} {r['ST_mean']:>9.3f} {r['ST_median']:>8.3f} {r['OT_mean']:>9.3f} {r['OT_median']:>8.3f}")

    # Per-backbone gap
    print("\nBackbone-wise CENTER GAP (UMCU - RUMC of ST):")
    for bb in BBS:
        if bb not in all_data: continue
        sts0 = [v["ST"] for v in all_data[bb].values() if v["center"] == 0]
        sts1 = [v["ST"] for v in all_data[bb].values() if v["center"] == 1]
        if not sts0 or not sts1: continue
        gap = np.mean(sts1) - np.mean(sts0)
        print(f"  {bb}: RUMC ST={np.mean(sts0):.3f} (n={len(sts0)}), UMCU ST={np.mean(sts1):.3f} (n={len(sts1)}), gap={gap:+.3f}")

    # Scatter: ST distribution per backbone, colored by center
    fig, axes = plt.subplots(1, len(BBS), figsize=(3.6 * len(BBS), 4.0), sharey=True)
    for j, bb in enumerate(BBS):
        ax = axes[j]
        if bb not in all_data:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes); continue
        d = all_data[bb]
        sids = sorted(d.keys(), key=lambda s: -d[s]["ST"])
        xs = np.arange(len(sids))
        colors = ["#1f77b4" if d[s]["center"] == 0 else "#d62728" for s in sids]
        ax.scatter(xs, [d[s]["ST"] for s in sids], c=colors, s=18, alpha=0.85)
        # medians per center
        sts0 = [d[s]["ST"] for s in sids if d[s]["center"] == 0]
        sts1 = [d[s]["ST"] for s in sids if d[s]["center"] == 1]
        ax.axhline(np.median(sts0), color="#1f77b4", linestyle="--", linewidth=1, alpha=0.7, label=f"RUMC med={np.median(sts0):.2f}")
        ax.axhline(np.median(sts1), color="#d62728", linestyle="--", linewidth=1, alpha=0.7, label=f"UMCU med={np.median(sts1):.2f}")
        ax.set_title(bb, fontsize=10)
        ax.set_xlabel("slide rank")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("per-slide ST (same-slide tumor neighbor frac)")
    fig.suptitle("CAM16 per-slide ST by center (blue=RUMC/0, red=UMCU/1)", fontsize=11)
    out = PSDIR / "per_slide_ST_by_center.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # Box plot ST and OT per center per backbone
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, metric in zip(axes, ["ST", "OT"]):
        data_by_center = {0: [], 1: []}
        labels = []
        for bb in BBS:
            if bb not in all_data: continue
            for c in (0, 1):
                vals = [v[metric] for v in all_data[bb].values() if v["center"] == c]
                data_by_center[c].append(vals)
            labels.append(bb)
        positions0 = np.arange(len(labels)) * 3
        positions1 = positions0 + 1
        bp0 = ax.boxplot(data_by_center[0], positions=positions0, widths=0.7, patch_artist=True,
                          showfliers=False, boxprops=dict(facecolor="#1f77b4", alpha=0.7))
        bp1 = ax.boxplot(data_by_center[1], positions=positions1, widths=0.7, patch_artist=True,
                          showfliers=False, boxprops=dict(facecolor="#d62728", alpha=0.7))
        ax.set_xticks(positions0 + 0.5)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_title("per-slide ST by center (blue=RUMC, red=UMCU)")
    axes[1].set_title("per-slide OT by center")
    out2 = PSDIR / "per_slide_box_by_center.png"
    fig.tight_layout()
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
