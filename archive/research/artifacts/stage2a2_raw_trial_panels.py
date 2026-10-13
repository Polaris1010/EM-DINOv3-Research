"""Re-render existing blinded A/B pairs from original EM pixels, without affinity."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parents[1]
REF = json.loads((RUN / "references/paths.json").read_text())
STAGE1 = Path(REF["stage1_root"])
STAGE2 = Path(REF["stage2_root"])
sys.path[:0] = [str(STAGE1 / "scripts"), str(STAGE1 / "topology_feasibility/mam_sam_vit_b")]
from mam_sam_vit_b_smoke import read_patch  # noqa: E402

OUT = RUN / "branch_S/blind_review_raw_20261011"


def select_trial(keys: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for key in keys:
        grouped[key["source"]].append(key)
    for group in grouped.values():
        group.sort(key=lambda k: k["pair_id"])
    chosen = [group[0] for _, group in sorted(grouped.items())]
    assert len(chosen) == 13
    chosen.extend(grouped["openorganelle"][1:5])
    for source, group in sorted(grouped.items()):
        if source != "openorganelle" and len(group) > 1 and len(chosen) < 20:
            chosen.append(group[1])
    assert len(chosen) == 20 and len({k["pair_id"] for k in chosen}) == 20
    return sorted(chosen, key=lambda k: k["pair_id"])


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    with (RUN / "branch_S/blind_review/answer_key_CONCEALED.csv").open(newline="") as stream:
        # Only identity and geometry fields are used; score values are not read.
        fields = ("pair_id", "sample_index", "source", "A_candidate", "B_candidate")
        keys = [{field: row[field] for field in fields} for row in csv.DictReader(stream)]
    assert len(keys) == 100
    with (STAGE2 / "topology_calibration/crop_score/crop_score_pilot_424x4.csv").open(newline="") as stream:
        geometry = {(int(r["sample_index"]), int(r["candidate"])): {k: int(r[k]) for k in ("x", "y", "w", "h")} for r in csv.DictReader(stream)}
    samples = json.loads((RUN / "references/sample_manifest_424.json").read_text())["samples"]
    panels, context = OUT / "panels", OUT / "context"
    panels.mkdir(parents=True, exist_ok=True)
    context.mkdir(exist_ok=True)
    validation = []
    for key in keys:
        i = int(key["sample_index"])
        rgb = read_patch(samples[i])
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
        assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])
        # Match the original audit's display geometry, including small sources.
        base = Image.fromarray(rgb).resize((512, 512))
        base.save(context / f"{key['pair_id']}.png", optimize=True)
        canvas = Image.new("RGB", (1048, 550), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        boxes = []
        for letter, offset in (("A", 0), ("B", 524)):
            row = geometry[i, int(key[f"{letter}_candidate"])]
            box = (row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"])
            boxes.append(row)
            view = base.copy()
            ImageDraw.Draw(view).rectangle(box, outline=(255, 220, 0), width=2)
            draw.text((offset + 10, 10), f"{key['pair_id']}  {letter}", fill="white")
            canvas.paste(view, (offset + 10, 30))
        assert (boxes[0]["w"], boxes[0]["h"]) == (boxes[1]["w"], boxes[1]["h"])
        canvas.save(panels / f"{key['pair_id']}.png", optimize=True)
        validation.append({"pair_id": key["pair_id"], "source": key["source"], "same_shape": True,
                           "raw_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                           "input_shape": list(rgb.shape), "display_shape": [512, 512, 3],
                           "display_rgb_sha256": hashlib.sha256(np.asarray(base).tobytes()).hexdigest()})
        if len(validation) % 20 == 0:
            print(f"raw panels {len(validation)}/100", flush=True)
    trial = select_trial(keys)
    with (OUT / "human_review_trial20.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair_id", "human_choice_A_B_unclear", "unclear_reason_optional", "note_optional"])
        writer.writeheader()
        for key in trial:
            writer.writerow({"pair_id": key["pair_id"]})
    with (OUT / "trial20_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair_id", "source"])
        writer.writeheader()
        writer.writerows({k: key[k] for k in ("pair_id", "source")} for key in trial)
    guide = OUT / "REVIEW_RULES_zh.md"
    assert guide.is_file(), "Upload review rules before packaging"
    with zipfile.ZipFile(OUT / "blind_raw_trial20.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in trial:
            pair_id = key["pair_id"]
            archive.write(panels / f"{pair_id}.png", f"panels/{pair_id}.png")
            archive.write(context / f"{pair_id}.png", f"context/{pair_id}.png")
        for name in ("human_review_trial20.csv", "trial20_manifest.csv", "REVIEW_RULES_zh.md"):
            archive.write(OUT / name, name)
    result = {"status": "ready_for_trial_review", "raw_panels": 100, "trial_pairs": 20,
              "sources_in_trial": 13, "trial_pair_ids": [key["pair_id"] for key in trial],
              "pairing_and_AB_order": "unchanged from original 100-pair audit",
              "selection": "one pair per source, four additional OpenOrganelle pairs, one additional pair from first three other sources with multiple pairs; ordered by existing pair ID; no visual or score selection",
              "pixels": "identical Stage1 normalized EM patch; no SAM overlay, score, or segmentation labels",
              "role": "clarify human judging rules; not a final S PASS/FAIL test",
              "zip_bytes": (OUT / "blind_raw_trial20.zip").stat().st_size,
              "validation": validation}
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "validation"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
