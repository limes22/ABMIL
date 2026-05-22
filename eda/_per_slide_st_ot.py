"""Per-slide ST/OT analysis. Reuses pools from manifold/{bb}_pools.npz.

For each backbone:
  - Strict positive pool (>=0.9, per-slide cap 200) + negatives (per-slide cap 200)
  - Run k-NN on combined pool
  - For each tumor query, get its 4 bucket counts
  - Aggregate per slide: mean ST, SN, OT, ON, cs_kNN, n_tumor_patches
  - Save CSV
  - Plot per-slide ST distribution

Then identify top/bottom-K slides by ST for targeted visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

POOL_DIR = Path("/workspace/_eda_lowdim_out/manifold")
OUT_DIR = POOL_DIR / "per_slide"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_NN = 11
THR_STRICT = 0.9
POS_PER_SLIDE = 200
NEG_PER_SLIDE = 200


def process(backbone: str):
    pool_path = POOL_DIR / f"{backbone}_pools.npz"
    if not pool_path.exists():
        print(f"[{backbone}] missing pool"); return None
    d = np.load(pool_path, allow_pickle=True)
    Xp_full = d["X_pos"]; sp_full = d["slides_pos"]; scores_p = d["scores_pos"]
    Xn_full = d["X_neg"]; sn_full = d["slides_neg"]

    rng = np.random.default_rng(0)
    pmask = scores_p >= THR_STRICT
    Xp = Xp_full[pmask]; sp = sp_full[pmask]
    # cap per slide
    keep = []
    for sid in np.unique(sp):
        idx = np.where(sp == sid)[0]
        if idx.size > POS_PER_SLIDE:
            idx = rng.choice(idx, POS_PER_SLIDE, replace=False)
        keep.extend(idx.tolist())
    keep = np.array(sorted(keep))
    Xp = Xp[keep]; sp = sp[keep]

    Xn = Xn_full; sn = sn_full
    keep_n = []
    for sid in np.unique(sn):
        idx = np.where(sn == sid)[0]
        if idx.size > NEG_PER_SLIDE:
            idx = rng.choice(idx, NEG_PER_SLIDE, replace=False)
        keep_n.extend(idx.tolist())
    keep_n = np.array(sorted(keep_n))
    Xn = Xn[keep_n]; sn = sn[keep_n]

    n_pos = Xp.shape[0]
    X_all = np.concatenate([Xp, Xn], axis=0)
    slides_all = np.concatenate([sp, sn])
    is_tumor = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(Xn.shape[0], dtype=bool)])
    print(f"[{backbone}] n_pos={n_pos}, n_neg={Xn.shape[0]}, fitting kNN...", flush=True)

    nn = NearestNeighbors(n_neighbors=K_NN, metric="euclidean").fit(X_all)
    _, nbr_idx = nn.kneighbors(X_all)

    # Compute buckets for tumor queries only
    buckets = np.zeros((n_pos, 4), dtype=np.float32)  # ST, SN, OT, ON
    for i in range(n_pos):
        for j in nbr_idx[i, 1:]:
            same = slides_all[j] == slides_all[i]
            nt = is_tumor[j]
            if same and nt: buckets[i, 0] += 1
            elif same and not nt: buckets[i, 1] += 1
            elif not same and nt: buckets[i, 2] += 1
            else: buckets[i, 3] += 1
    buckets /= 10.0

    # Aggregate per slide
    slide_stats = {}
    for sid in np.unique(sp):
        m = sp == sid
        b = buckets[m].mean(0)
        slide_stats[sid] = {
            "n_tumor_patches": int(m.sum()),
            "ST": float(b[0]),
            "SN": float(b[1]),
            "OT": float(b[2]),
            "ON": float(b[3]),
            "cs_kNN": float(b[2] + b[3]),
        }

    # Sort by ST descending (worst = most slide-bound) and save CSV
    rows = sorted(slide_stats.items(), key=lambda kv: -kv[1]["ST"])
    out_csv = OUT_DIR / f"{backbone}_per_slide.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide_id", "n_tumor_patches", "ST", "SN", "OT", "ON", "cs_kNN"])
        for sid, st in rows:
            w.writerow([sid, st["n_tumor_patches"], f"{st['ST']:.4f}", f"{st['SN']:.4f}",
                        f"{st['OT']:.4f}", f"{st['ON']:.4f}", f"{st['cs_kNN']:.4f}"])

    print(f"[{backbone}] saved {out_csv}", flush=True)
    print(f"  ST per-slide: mean={np.mean([s['ST'] for s in slide_stats.values()]):.3f}, "
          f"median={np.median([s['ST'] for s in slide_stats.values()]):.3f}, "
          f"p25={np.quantile([s['ST'] for s in slide_stats.values()], 0.25):.3f}, "
          f"p75={np.quantile([s['ST'] for s in slide_stats.values()], 0.75):.3f}", flush=True)
    print(f"  worst-10 (highest ST):", flush=True)
    for sid, st in rows[:10]:
        print(f"    {sid}: ST={st['ST']:.3f}, OT={st['OT']:.3f}, n={st['n_tumor_patches']}")
    print(f"  best-10 (lowest ST):", flush=True)
    for sid, st in rows[-10:]:
        print(f"    {sid}: ST={st['ST']:.3f}, OT={st['OT']:.3f}, n={st['n_tumor_patches']}")

    return slide_stats


def plot_distribution(backbones, results):
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, bb in enumerate(backbones):
        if bb not in results: continue
        sts = sorted([(sid, s["ST"]) for sid, s in results[bb].items()], key=lambda x: -x[1])
        ys = [v for _, v in sts]
        ax.plot(range(len(ys)), ys, label=bb, marker="o", markersize=2, linewidth=1.2)
    ax.set_xlabel("slide rank (by ST, highest first)")
    ax.set_ylabel("per-slide ST (same-slide tumor neighbor fraction)")
    ax.set_title("Per-slide slide-binding strength (ST), sorted desc — flat-high = pervasive batch, steep drop = localized")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    out = OUT_DIR / "per_slide_ST_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Histogram per backbone
    fig, axes = plt.subplots(1, len(backbones), figsize=(3.6 * len(backbones), 3.6))
    for i, bb in enumerate(backbones):
        ax = axes[i] if len(backbones) > 1 else axes
        if bb not in results:
            ax.text(0.5, 0.5, "missing"); continue
        sts = [s["ST"] for s in results[bb].values()]
        ax.hist(sts, bins=20, range=(0, 1), edgecolor="black", linewidth=0.5)
        ax.axvline(np.median(sts), color="red", linestyle="--", linewidth=1, label=f"median={np.median(sts):.2f}")
        ax.set_title(bb)
        ax.set_xlabel("per-slide ST")
        ax.set_ylim(0, 60)
        ax.legend(fontsize=8)
    out2 = OUT_DIR / "per_slide_ST_hist.png"
    fig.tight_layout()
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="resnet,uni,virchow")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    if args.plot_only:
        # Load CSVs back into results dict
        import csv as _csv
        results = {}
        for bb in backbones:
            p = OUT_DIR / f"{bb}_per_slide.csv"
            if not p.exists():
                print(f"  missing {p}"); continue
            results[bb] = {}
            with open(p) as f:
                r = _csv.DictReader(f)
                for row in r:
                    results[bb][row["slide_id"]] = {
                        "ST": float(row["ST"]),
                        "OT": float(row["OT"]),
                        "n_tumor_patches": int(row["n_tumor_patches"]),
                    }
        plot_distribution(backbones, results)
        return

    results = {}
    for bb in backbones:
        r = process(bb)
        if r is not None:
            results[bb] = r
    plot_distribution(backbones, results)


if __name__ == "__main__":
    main()
