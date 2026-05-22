#!/usr/bin/env python3
"""Fixed regex — Vanilla folders use 4 or 6 digit time."""
import os, re, glob

RESULTS = "/workspace/results"
FEAT_BASE = "/workspace/features"

DATASETS = ["BRACS", "CAM16", "CAM17"]
BACKBONES = ["CONCH", "Virchow", "UNI"]
HEADS = ["ABMIL", "CLAM", "ACMIL"]

FEAT_DIRS = {
    ("BRACS","CONCH"): ["bracs_conch"], ("BRACS","Virchow"): ["bracs_virchow"], ("BRACS","UNI"): ["bracs_uni"],
    ("CAM16","CONCH"): ["cam16_conch","camelyon16_conch_features"],
    ("CAM16","Virchow"): ["cam16_virchow","camelyon16_virchow_features"],
    ("CAM16","UNI"): ["cam16_uni","camelyon16_uni_features"],
    ("CAM17","CONCH"): ["cam17_conch_features"],
    ("CAM17","Virchow"): ["cam17_virchow_features"],
    ("CAM17","UNI"): ["cam17_uni_features"],
}

def feat_exists(ds, fe):
    for c in FEAT_DIRS.get((ds, fe), []):
        if os.path.isdir(f"{FEAT_BASE}/{c}"): return c
    return None

def has_finished_split(folder_pattern):
    seeds = set()
    for d in glob.glob(f"{RESULTS}/{folder_pattern}"):
        m = re.search(r"_s(\d+)_\d{8}_\d{4,6}_s\d+/?$", d)
        if not m: continue
        if os.path.exists(f"{d}/split_0_results.pkl"):
            seeds.add(int(m.group(1)))
    return sorted(seeds)

print(f"{'ds':<6} {'fe':<8} {'head':<6} {'feat':<25} {'ECCDI5':<18} {'Vanilla':<18}")
print("-" * 90)
for ds in DATASETS:
    for fe in BACKBONES:
        feat = feat_exists(ds, fe)
        for head in HEADS:
            ecc = has_finished_split(f"{ds}_{fe}_{head}_ECCDI5_s*")
            van = has_finished_split(f"{ds}_{fe}_{head}_Vanilla_s*")
            print(f"{ds:<6} {fe:<8} {head:<6} {str(feat):<25} {str(ecc):<18} {str(van):<18}")
