"""Generate paired raw-EM and frozen SAM/OrgMIM MAM augmentation maps on CPU."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from mam_sam_vit_b_smoke import EXPECTED_SHA, MAM_DIR, TOP, WEIGHT, mam_from_rgb, read_patch, sha256


NAMES = ["identity", "flip_horizontal", "flip_vertical", "weak_jitter", "strong_jitter", "weak_blur", "strong_blur", "mild_crop", "medium_crop", "aggressive_crop"]
PAIRED = MAM_DIR / "paired"
MAPS = PAIRED / "maps"


def augment(rgb: np.ndarray) -> list[np.ndarray]:
    def gamma(power: float, offset: float) -> np.ndarray:
        x = np.clip((rgb.astype(np.float32) / 255.0) ** power + offset, 0, 1)
        return (255 * x).astype(np.uint8)

    def crop(fraction: float) -> np.ndarray:
        side = round(rgb.shape[0] * fraction)
        start = (rgb.shape[0] - side) // 2
        return cv2.resize(rgb[start:start + side, start:start + side], rgb.shape[:2][::-1], interpolation=cv2.INTER_AREA)

    return [
        rgb, np.ascontiguousarray(rgb[:, ::-1]), np.ascontiguousarray(rgb[::-1, :]),
        gamma(.9, .02), gamma(.55, .10),
        cv2.GaussianBlur(rgb, (0, 0), .5), cv2.GaussianBlur(rgb, (0, 0), 2.0),
        crop(.85), crop(.65), crop(.45),
    ]


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=424)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
    assert sha256(WEIGHT) == EXPECTED_SHA
    torch.set_num_threads(4)
    MAPS.mkdir(parents=True, exist_ok=True)
    samples = json.loads((TOP / "sample_manifest.json").read_text())["samples"]
    assert len(samples) == 424 and 1 <= args.limit <= len(samples)
    model = sam_model_registry["vit_b"](checkpoint=str(WEIGHT)).to("cpu").eval()
    predictor = SamPredictor(model)
    started = time.monotonic()
    generated = 0
    for index, sample in enumerate(samples[:args.limit]):
        target = MAPS / f"sample_{index:06d}.npz"
        if target.exists():
            continue
        t0 = time.monotonic()
        rgb = read_patch(sample)
        mams, raws = [], []
        for variant in augment(rgb):
            mam, _ = mam_from_rgb(predictor, variant)
            mams.append(mam)
            raw = cv2.resize(variant[:, :, 0], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            raws.append(raw)
        mam_array = np.stack(mams)
        raw_array = np.stack(raws)
        assert mam_array.shape == raw_array.shape == (10, 64, 64)
        assert np.isfinite(mam_array).all()
        temp = MAPS / f"sample_{index:06d}.tmp.npz"
        np.savez_compressed(temp, mam=mam_array, raw=raw_array)
        temp.replace(target)
        generated += 1
        atomic_json(PAIRED / "generation_progress.json", {
            "completed_sample_index": index, "total_samples": len(samples),
            "files_present": sum(1 for _ in MAPS.glob("sample_*.npz")),
            "generated_this_process": generated, "last_sample_seconds": time.monotonic() - t0,
            "elapsed_seconds": time.monotonic() - started,
            "last_volume_id": sample["volume_id"], "model_sha256": EXPECTED_SHA,
            "cpu_only": True, "uses_labels": False,
        })
    if args.limit == len(samples):
        files = sorted(MAPS.glob("sample_*.npz"))
        assert len(files) == len(samples)
        atomic_json(PAIRED / "generation_complete.json", {
            "status": "complete", "sample_count": len(samples), "augmentation_names": NAMES,
            "model": "Meta SAM ViT-B 01ec64 frozen CPU", "model_sha256": EXPECTED_SHA,
            "affinity": "OrgMIM min of vertical and horizontal embeddings_to_affinities; delta_v=.5, delta_d=1.5",
            "uses_labels": False, "gpu_used": False,
        })
    print(json.dumps({"limit": args.limit, "new_files": generated, "complete": (PAIRED / "generation_complete.json").exists()}))


if __name__ == "__main__":
    main()
