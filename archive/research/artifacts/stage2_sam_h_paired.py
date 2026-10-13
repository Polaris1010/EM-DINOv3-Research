"""Frozen SAM ViT-H MAMs for the Stage 1 fixed 424-slice sample."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


RUN = Path(__file__).resolve().parents[1]
STAGE1 = Path(json.loads((RUN / "stage1_reference/stage1_paths.json").read_text())["stage1_root"])
sys.path.insert(0, str(STAGE1 / "topology_feasibility/mam_sam_vit_b"))
sys.path.insert(0, str(STAGE1 / "scripts"))
from mam_sam_vit_b_smoke import mam_from_rgb, read_patch, sha256  # noqa: E402

WEIGHT = RUN / "topology_calibration/sam_vit_h/sam_vit_h_4b8939.pth"
EXPECTED_SHA = "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
OUT = RUN / "topology_calibration/sam_vit_h"
MAPS = OUT / "paired_maps"
NAMES = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")


def transforms(rgb: np.ndarray) -> list[np.ndarray]:
    return [rgb, np.flip(rgb, 1).copy(), np.flip(rgb, 0).copy(),
            np.rot90(rgb, 1).copy(), np.rot90(rgb, 2).copy(), np.rot90(rgb, 3).copy()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=424)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    assert 0 <= args.shard < args.num_shards <= 3
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"1", "2", "3"}
    assert json.loads((RUN / "audit/sample_validation.json").read_text())["status"] == "pass"
    assert WEIGHT.stat().st_size == 2564550879 and sha256(WEIGHT) == EXPECTED_SHA
    torch.set_num_threads(4)
    samples = json.loads((RUN / "topology_calibration/sample_manifest.json").read_text())["samples"]
    assert len(samples) == 424 and 1 <= args.limit <= 424
    MAPS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    model = sam_model_registry["vit_h"](checkpoint=str(WEIGHT)).to(device).eval()
    predictor = SamPredictor(model)
    t_start = time.monotonic()
    new_count = 0
    for index, sample in enumerate(samples[:args.limit]):
        if index % args.num_shards != args.shard:
            continue
        target = MAPS / f"sample_{index:06d}.npz"
        if target.exists():
            with np.load(target) as package:
                assert package["mam"].shape == (6, 64, 64)
            continue
        t0 = time.monotonic()
        rgb = read_patch(sample)
        maps = []
        for variant in transforms(rgb):
            mam, embed_shape = mam_from_rgb(predictor, variant)
            assert mam.shape == (64, 64) and np.isfinite(mam).all()
            maps.append(mam)
        data = np.stack(maps)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, mam=data)
        temporary.replace(target)
        new_count += 1
        report = {"status": "running", "sample_index": index, "files_present": len(list(MAPS.glob("sample_*.npz"))),
                  "last_sample_seconds": time.monotonic() - t0, "elapsed_seconds": time.monotonic() - t_start,
                  "mean_seconds_per_sample": (time.monotonic() - t_start) / new_count,
                  "embedding_shape": embed_shape, "weight_sha256": EXPECTED_SHA,
                  "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "model": "SAM ViT-H frozen", "uses_labels": False}
        (OUT / f"generation_progress_shard{args.shard}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    if args.limit == 424 and len(list(MAPS.glob("sample_*.npz"))) == 424:
        (OUT / "generation_complete.json").write_text(json.dumps({"status": "complete", "samples": 424,
                "transforms": NAMES, "weight_sha256": EXPECTED_SHA, "uses_labels": False}, indent=2))


if __name__ == "__main__":
    main()
