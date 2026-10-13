"""Summarize paired raw-EM versus frozen SAM/OrgMIM MAM PH distributions."""
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
PAIRED = TOP / "mam_sam_vit_b" / "paired"
FIG = PAIRED / "figures"
FIG.mkdir(exist_ok=True)
NAMES = ["identity", "flip_horizontal", "flip_vertical", "weak_jitter", "strong_jitter", "weak_blur", "strong_blur", "mild_crop", "medium_crop", "aggressive_crop"]
MODES = ["raw_em_paired", "mam_affinity", "mam_boundary"]


def quant(values, p):
    return float(np.quantile(np.asarray(values), p))


def main() -> None:
    assert json.loads((PAIRED / "ph_complete.json").read_text())["status"] == "complete"
    rows = list(csv.DictReader((PAIRED / "metrics_paired.csv").open()))
    assert len(rows) == 424 * 10 * 3
    grouped = defaultdict(list)
    sources = set()
    for row in rows:
        for key in ("d0", "d1"):
            row[key] = float(row[key])
        row["d_equal"] = row["d0"] + row["d1"]
        grouped[(row["input"], row["augmentation"])].append(row)
        sources.add(row["source"])
    assert len(sources) == 13
    summary = {"scope": "paired raw EM and frozen SAM ViT-B OrgMIM affinity MAM on same augmented slices; no labels", "samples": 424, "sources": 13, "rows": len(rows), "distance": "approximate bottleneck e=0.001; H0/H1 lifetime >=0.02", "equal_sum": "d0+d1 exploratory, not a selected alpha/beta", "modes": {}, "source_medians": {}}
    for mode in MODES:
        summary["modes"][mode] = {}
        for name in NAMES:
            group = grouped[(mode, name)]
            assert len(group) == 424
            summary["modes"][mode][name] = {key: {"p10": quant([r[key] for r in group], .1), "p50": quant([r[key] for r in group], .5), "p90": quant([r[key] for r in group], .9)} for key in ("d0", "d1", "d_equal")}
        pairs = defaultdict(dict)
        for row in rows:
            if row["input"] == mode:
                pairs[row["sample_index"]][row["augmentation"]] = row["d_equal"]
        summary["modes"][mode]["pairwise_fraction"] = {
            "aggressive_crop_gt_mild_crop": sum(v["aggressive_crop"] > v["mild_crop"] for v in pairs.values()) / 424,
            "strong_blur_gt_weak_blur": sum(v["strong_blur"] > v["weak_blur"] for v in pairs.values()) / 424,
            "strong_jitter_gt_weak_jitter": sum(v["strong_jitter"] > v["weak_jitter"] for v in pairs.values()) / 424,
        }
        for key in ("d0", "d1", "d_equal"):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.boxplot([[r[key] for r in grouped[(mode, name)]] for name in NAMES], tick_labels=NAMES, showfliers=False)
            ax.set_title(f"{mode}: {key} on 424 paired EM slices")
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            fig.savefig(FIG / f"{mode}_{key}.png", dpi=160)
            plt.close(fig)
        summary["source_medians"][mode] = {}
        for source in sorted(sources):
            summary["source_medians"][mode][source] = {name: quant([r["d_equal"] for r in grouped[(mode, name)] if r["source"] == source], .5) for name in NAMES}
        matrix = np.asarray([[summary["source_medians"][mode][source][name] for name in NAMES] for source in sorted(sources)])
        fig, ax = plt.subplots(figsize=(12, 5.5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(sources)), sorted(sources), fontsize=7)
        ax.set_xticks(range(len(NAMES)), NAMES, rotation=35, ha="right")
        ax.set_title(f"{mode}: source median d0+d1")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIG / f"{mode}_source_heatmap.png", dpi=160)
        plt.close(fig)
    (PAIRED / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Paired raw EM and SAM MAM PH feasibility", "", "424 label-free slices from all 212 volumes and 13 sources; each of 10 augmentations was applied to the same image before raw filtration and frozen SAM ViT-B feature extraction. MAM used OrgMIM's official affinity function (delta_v=0.5, delta_d=1.5; min of vertical and horizontal affinities). Both affinity and inverse boundary filtrations were tested. This is a ViT-B feasibility run; OrgMIM's README example uses ViT-H, so results should not be described as ViT-H replication.", "", "Distances are approximate bottleneck values (e=0.001) after H0/H1 lifetime >=0.02 filtering. Equal sum is an exploratory display; it is not a tuned TopoAug score.", "", "| Input | Augmentation | H0 p50 | H1 p50 | Sum p50 | Sum p90 |", "|---|---|---:|---:|---:|---:|"]
    for mode in MODES:
        for name in NAMES:
            v = summary["modes"][mode][name]
            lines.append(f"| {mode} | {name} | {v['d0']['p50']:.3f} | {v['d1']['p50']:.3f} | {v['d_equal']['p50']:.3f} | {v['d_equal']['p90']:.3f} |")
    lines += ["", "## Paired ordering checks", "", "Fractions below show the share of slices whose strong augmentation distance exceeded its mild counterpart; they measure dynamic range, not biological correctness.", "", "| Input | Aggressive crop > mild crop | Strong blur > weak blur | Strong jitter > weak jitter |", "|---|---:|---:|---:|"]
    for mode in MODES:
        p = summary["modes"][mode]["pairwise_fraction"]
        lines.append(f"| {mode} | {p['aggressive_crop_gt_mild_crop']:.3f} | {p['strong_blur_gt_weak_blur']:.3f} | {p['strong_jitter_gt_weak_jitter']:.3f} |")
    lines += ["", "## Interpretation", "", "SAM feature maps may change under flips because the encoder is not strictly flip equivariant. Intensity jitter changes filtration values even if geometry is unchanged. Crops change field of view. These distances alone cannot prove mitochondrial topology preservation or damage. Inspect source-stratified distributions and representative map images before selecting alpha, beta, or a threshold. No GT mask was read or used.", "", "Figures and numeric summary are in this directory. The DINOv3 B_full training environment and GPU assignments were not changed."]
    (PAIRED / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "report": str(PAIRED / 'report.md')}))


if __name__ == "__main__":
    main()
