"""Prepare concealed 100-pair crop audit without exposing score order on images."""
from __future__ import annotations

import csv
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN = Path(__file__).resolve().parents[1]
REF = json.loads((RUN / "references/paths.json").read_text())
STAGE1 = Path(REF["stage1_root"])
sys.path[:0] = [str(STAGE1 / "scripts"), str(STAGE1 / "topology_feasibility/mam_sam_vit_b")]
from mam_sam_vit_b_smoke import read_patch  # noqa: E402

OUT = RUN / "branch_S/blind_review"
PANELS = OUT / "panels"


def choose_pairs(rows: list[dict]) -> list[dict]:
    by_sample = defaultdict(list)
    for row in rows:
        by_sample[int(row["sample_index"])].append(row)
    eligible = defaultdict(list)
    for i, group in by_sample.items():
        full = [r for r in group if r["all_four_sides_observed"] == "True" and np.isfinite(float(r["cut_mean"]))]
        if len(full) < 2:
            continue
        full.sort(key=lambda r: float(r["cut_mean"]))
        margin = float(full[-1]["cut_mean"]) - float(full[0]["cut_mean"])
        if margin <= 1e-6:
            continue
        eligible[group[0]["source"]].append({"sample_index": i, "source": group[0]["source"], "low": full[0], "high": full[-1], "margin": margin})
    rng = random.Random(20261002)
    for group in eligible.values():
        rng.shuffle(group)
    selected = eligible.pop("openorganelle")[:50]
    source_names = sorted(eligible)
    while len(selected) < 100:
        changed = False
        for source in source_names:
            if eligible[source] and len(selected) < 100:
                selected.append(eligible[source].pop())
                changed = True
        if not changed:
            raise RuntimeError("Insufficient review pairs")
    rng.shuffle(selected)
    for index, item in enumerate(selected, 1):
        order = [item["low"], item["high"]]
        rng.shuffle(order)
        item["pair_id"] = f"P{index:03d}"
        item["A"], item["B"] = order
    return selected


def panel(sample: dict, item: dict, ay: np.ndarray, ax: np.ndarray) -> Image.Image:
    rgb = read_patch(sample)
    base = Image.fromarray(rgb, "RGB").resize((512, 512))
    # Blue/cyan affinity overlay is identical in both candidate views.
    affinity = np.minimum(ay, ax)
    heat = np.zeros((64, 64, 4), dtype=np.uint8)
    heat[..., 1] = (200 * affinity).astype(np.uint8)
    heat[..., 2] = (255 * affinity).astype(np.uint8)
    heat[..., 3] = (90 * affinity).astype(np.uint8)
    overlay = Image.fromarray(heat, "RGBA").resize((512, 512), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1048, 550), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"{item['pair_id']}  A", fill="white")
    draw.text((534, 10), f"{item['pair_id']}  B", fill="white")
    for offset, row in ((0, item["A"]), (524, item["B"])):
        view = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        viewdraw = ImageDraw.Draw(view)
        box = (int(row["x"]), int(row["y"]), int(row["x"]) + int(row["w"]), int(row["y"]) + int(row["h"]))
        viewdraw.rectangle(box, outline=(255, 220, 0), width=4)
        canvas.paste(view, (offset + 10, 30))
    return canvas


def main() -> None:
    samples = json.loads((RUN / "references/sample_manifest_424.json").read_text())["samples"]
    with (RUN / "branch_S/graph_cut/same_shape_scores_424.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    stage2 = Path(REF["stage2_root"])
    with (stage2 / "topology_calibration/crop_score/crop_score_pilot_424x4.csv").open(newline="") as stream:
        geometry = {(int(r["sample_index"]), int(r["candidate"])): r for r in csv.DictReader(stream)}
    for row in rows:
        row.update({k: geometry[int(row["sample_index"]), int(row["candidate"])][k] for k in ("x", "y", "w", "h")})
    chosen = choose_pairs(rows)
    PANELS.mkdir(parents=True, exist_ok=True)
    answer_key = []
    blank = []
    for item in chosen:
        i = item["sample_index"]
        with np.load(RUN / f"branch_S/directional_affinity/maps/sample_{i:06d}.npz") as data:
            im = panel(samples[i], item, data["ay"], data["ax"])
        im.save(PANELS / f"{item['pair_id']}.png", optimize=True)
        answer_key.append({"pair_id": item["pair_id"], "sample_index": i, "source": item["source"],
                           "A_candidate": item["A"]["candidate"], "B_candidate": item["B"]["candidate"],
                           "score_winner": "A" if item["A"] is item["low"] else "B",
                           "score_A": item["A"]["cut_mean"], "score_B": item["B"]["cut_mean"],
                           "margin": item["margin"]})
        blank.append({"pair_id": item["pair_id"], "human_choice_A_B_unclear": ""})
        if len(answer_key) % 20 == 0:
            print(f"panels {len(answer_key)}/100", flush=True)
    for name, data in (("answer_key_CONCEALED.csv", answer_key), ("human_review_blank.csv", blank)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    with zipfile.ZipFile(OUT / "blind_panels_100.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PANELS.glob("*.png")):
            archive.write(path, arcname=path.name)
        archive.write(OUT / "human_review_blank.csv", arcname="human_review_blank.csv")
    manifest = {"status": "ready_for_human_review", "pairs": 100, "images": "panels/P001.png...P100.png",
                "selection": "50 openorganelle, 50 round-robin other sources; highest vs lowest cut_mean among candidates with all four crossing boundaries observed",
                "candidate_order": "A/B shuffled with seed 20261002", "metric_locked_before_review": "cut_mean",
                "zip_bytes": (OUT / "blind_panels_100.zip").stat().st_size}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
