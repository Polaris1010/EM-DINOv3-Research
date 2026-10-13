"""Summarize the label-free raw EM PH diagnostic without claiming MAM validation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "topology_feasibility"
FIG = TOP / "figures"
FIG.mkdir(exist_ok=True)


def q(values, p):
    return float(np.quantile(np.asarray(values), p))


rows = list(csv.DictReader((TOP / "metrics.csv").open()))
for r in rows:
    r["d0"] = float(r["d0"])
    r["d1"] = float(r["d1"])
    r["d_equal"] = r["d0"] + r["d1"]
names = ["identity", "flip_horizontal", "flip_vertical", "weak_jitter", "strong_jitter", "weak_blur", "strong_blur", "mild_crop", "medium_crop", "aggressive_crop"]
assert len(rows) == 424 * len(names), len(rows)
sources = sorted({r["source"] for r in rows})
assert len(sources) == 13, sources

by_aug = defaultdict(list)
by_source = defaultdict(list)
for r in rows:
    by_aug[r["augmentation"]].append(r)
    by_source[(r["source"], r["augmentation"])].append(r)

summary = {
    "scope": "raw EM only; OrgMIM MAM unavailable and untested",
    "sample_count": 424,
    "source_count": 13,
    "rows": len(rows),
    "grid_size": 64,
    "min_lifetime": 0.02,
    "distance_approximation_e": 0.001,
    "combined_distance": "d0+d1, exploratory equal weights only",
    "augmentations": {},
    "sources": {},
}
for name in names:
    group = by_aug[name]
    summary["augmentations"][name] = {key: {"p10": q([r[key] for r in group], .1), "p50": q([r[key] for r in group], .5), "p90": q([r[key] for r in group], .9)} for key in ("d0", "d1", "d_equal")}
for source in sources:
    summary["sources"][source] = {name: {key: q([r[key] for r in by_source[(source, name)]], .5) for key in ("d0", "d1", "d_equal")} for name in names}
(TOP / "summary.json").write_text(json.dumps(summary, indent=2))

for key, title in (("d0", "H0 bottleneck distance"), ("d1", "H1 bottleneck distance"), ("d_equal", "H0 + H1, exploratory")):
    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.boxplot([[r[key] for r in by_aug[name]] for name in names], tick_labels=names, showfliers=False)
    ax.set_ylabel(title)
    ax.set_title("Raw EM PH distance across 424 label-free slices")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG / f"raw_{key}_by_augmentation.png", dpi=160)
    plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, key in zip(axes, ("d0", "d1")):
    matrix = np.asarray([[summary["sources"][src][aug][key] for aug in names] for src in sources])
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(sources)), sources, fontsize=7)
    ax.set_ylabel("Source")
    ax.set_title(f"Median {key} by source")
    fig.colorbar(im, ax=ax, fraction=.02)
axes[-1].set_xticks(range(len(names)), names, rotation=35, ha="right")
fig.tight_layout()
fig.savefig(FIG / "raw_source_heatmap.png", dpi=160)
plt.close(fig)

lines = ["# Stage 1 topology feasibility: raw EM diagnostic", "", "This is a label-free exploratory result on 424 slices (2 per each of 212 volumes; 13 sources). It does not establish mitochondrial topology or validate a structure prior. OrgMIM MAM and the raw-vs-MAM comparison remain unavailable because the SAM/OrgMIM checkpoint and code were not present on the server and external repository access failed.", "", "Method: center 512-pixel-or-smaller square per selected Z slice, robust 1st/99th percentile normalization, resize to 64×64, cubical sublevel PH, H0/H1 intervals with lifetime ≥0.02, approximate bottleneck distance e=0.001. Crop variants are resized back to 64×64. Exact flip invariance is within the bottleneck solver tolerance. The equal-weight sum d0+d1 is an exploratory display only; no alpha/beta or production threshold is selected.", "", "| Augmentation | H0 p50 | H0 p90 | H1 p50 | H1 p90 | Sum p50 |", "|---|---:|---:|---:|---:|---:|"]
for name in names:
    v = summary["augmentations"][name]
    lines.append(f"| {name} | {v['d0']['p50']:.3f} | {v['d0']['p90']:.3f} | {v['d1']['p50']:.3f} | {v['d1']['p90']:.3f} | {v['d_equal']['p50']:.3f} |")
lines += ["", "## Interpretation limits", "", "Intensity jitter changes filtration values even if geometry is preserved; its bottleneck distance is not direct evidence of topological damage. Cropping changes field of view and can also cause distance without mitochondrial damage. These distances need comparison with frozen MAM and visual structure checks before any TopoAug gate is used.", "", "## Stage 2 readiness", "", "GUDHI is installed only under `topology_feasibility/vendor`, separate from the DINO training environment. The raw PH pipeline and source-stratified figures are ready. Recover an official OrgMIM-compatible SAM checkpoint/code, generate MAM on the same 424 image-only slices, then repeat this exact augmentation and PH analysis. Compare within-source weak/strong distributions and choose a quantile threshold only after MAM confirms separation. Do not use GT masks or label-selected crops.", "", "Figures: `figures/raw_d0_by_augmentation.png`, `raw_d1_by_augmentation.png`, `raw_d_equal_by_augmentation.png`, and `raw_source_heatmap.png`."]
(TOP / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"rows": len(rows), "sources": len(sources), "report": str(TOP / 'report.md')}))
