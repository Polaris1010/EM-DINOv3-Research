"""Label-free raw-EM PH diagnostic for the Stage 1 topology sample manifest."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import gudhi
import numpy as np
import scipy.ndimage as ndi
import zarr
from skimage.transform import resize


ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "topology_feasibility"
SIZE = 64
MIN_LIFETIME = 0.02


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def image_for_sample(sample: dict) -> np.ndarray:
    arr = zarr.open(sample["local_path"], mode="r")[sample["image_array_path"]]
    z = int(sample["slice_index"])
    h, w = map(int, sample["image_shape"])
    side = min(h, w, 512)
    y = (h - side) // 2
    x = (w - side) // 2
    patch = np.asarray(arr[z, y : y + side, x : x + side])
    return resize(normalize(patch), (SIZE, SIZE), anti_aliasing=True, preserve_range=True).astype(np.float32)


def centered_crop(x: np.ndarray, fraction: float) -> np.ndarray:
    side = round(SIZE * fraction)
    start = (SIZE - side) // 2
    return resize(x[start : start + side, start : start + side], (SIZE, SIZE), anti_aliasing=True, preserve_range=True).astype(np.float32)


def augmentations(x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": x,
        "flip_horizontal": x[:, ::-1],
        "flip_vertical": x[::-1, :],
        "weak_jitter": np.clip(x**0.9 + 0.02, 0, 1),
        "strong_jitter": np.clip(x**0.55 + 0.10, 0, 1),
        "weak_blur": ndi.gaussian_filter(x, 0.5),
        "strong_blur": ndi.gaussian_filter(x, 2.0),
        "mild_crop": centered_crop(x, 0.85),
        "medium_crop": centered_crop(x, 0.65),
        "aggressive_crop": centered_crop(x, 0.45),
    }


def diagrams(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cc = gudhi.CubicalComplex(top_dimensional_cells=np.ascontiguousarray(x).ravel(), dimensions=x.shape)
    cc.persistence(min_persistence=0)
    output = []
    for dim in (0, 1):
        intervals = np.asarray(cc.persistence_intervals_in_dimension(dim), dtype=np.float64).reshape(-1, 2)
        intervals = intervals[np.isfinite(intervals).all(axis=1)]
        intervals = intervals[intervals[:, 1] - intervals[:, 0] >= MIN_LIFETIME]
        output.append(intervals)
    return output[0], output[1]


def distance(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) and not len(b):
        return 0.0
    return float(gudhi.bottleneck_distance(a, b, e=0.001))


def main() -> None:
    manifest = json.loads((TOP / "sample_manifest.json").read_text())
    samples = manifest["samples"]
    assert len(samples) == 424
    path = TOP / "metrics.csv"
    columns = ["volume_id", "source", "slice_index", "input", "augmentation", "d0", "d1", "h0_count", "h1_count", "h0_aug_count", "h1_aug_count"]
    started = time.time()
    failures = []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for index, sample in enumerate(samples):
            try:
                x = image_for_sample(sample)
                original = diagrams(x)
                for name, aug in augmentations(x).items():
                    current = diagrams(aug)
                    writer.writerow({
                        "volume_id": sample["volume_id"], "source": sample["source"],
                        "slice_index": sample["slice_index"], "input": "raw_em", "augmentation": name,
                        "d0": distance(original[0], current[0]), "d1": distance(original[1], current[1]),
                        "h0_count": len(original[0]), "h1_count": len(original[1]),
                        "h0_aug_count": len(current[0]), "h1_aug_count": len(current[1]),
                    })
                f.flush()
            except Exception as exc:
                failures.append({"index": index, "volume_id": sample["volume_id"], "error": repr(exc)})
            if (index + 1) % 20 == 0 or index + 1 == len(samples):
                (TOP / "raw_ph_progress.json").write_text(json.dumps({"completed_samples": index + 1, "total_samples": len(samples), "failures": len(failures), "elapsed_seconds": time.time() - started}, indent=2))
    (TOP / "raw_ph_failures.json").write_text(json.dumps(failures, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} sample failures; see raw_ph_failures.json")


if __name__ == "__main__":
    main()
