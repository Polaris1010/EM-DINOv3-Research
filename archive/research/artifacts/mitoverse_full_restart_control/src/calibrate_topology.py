"""Offline 2D cubical-PH calibration on all MitoVerse images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import gudhi
import numpy as np
import zarr
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from full_manifest import validate_manifest


# All candidates are non-identity and within the v2 appearance bounds.
CANDIDATES = [
    {"name": "base", "gamma": [0.8, 1.2], "contrast": [0.8, 1.2], "noise": 0.01, "blur": 0.3},
    {"name": "gamma_mild", "gamma": [0.8, 1.2]},
    {"name": "gamma_broad", "gamma": [0.6, 1.4]},
    {"name": "contrast_mild", "contrast": [0.8, 1.2]},
    {"name": "noise_001", "noise": 0.01},
    {"name": "noise_002", "noise": 0.02},
    {"name": "noise_003", "noise": 0.03},
    {"name": "blur_03", "blur": 0.3},
    {"name": "blur_06", "blur": 0.6},
    {"name": "blur_10", "blur": 1.0},
    {"name": "gamma_noise", "gamma": [0.8, 1.2], "noise": 0.02},
    {"name": "contrast_blur", "contrast": [0.8, 1.2], "blur": 0.6},
]


def appearance(image: np.ndarray, candidate: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32, copy=True)
    if "gamma" in candidate:
        low, high = candidate["gamma"]
        x = np.power(np.clip(x, 0, 1), float(rng.uniform(low, high)))
    if "contrast" in candidate:
        low, high = candidate["contrast"]
        factor = float(rng.uniform(low, high))
        x = (x - float(x.mean())) * factor + float(x.mean())
    if "blur" in candidate:
        x = gaussian_filter(x, sigma=float(candidate["blur"]), mode="reflect")
    if "noise" in candidate:
        x = x + rng.normal(0.0, float(candidate["noise"]), size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def downsample(image: np.ndarray) -> np.ndarray:
    return resize(image, (64, 64), order=1, mode="reflect", anti_aliasing=True,
                  preserve_range=True).astype(np.float64)


def diagrams(image64: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=image64)
    pairs = complex_.persistence(homology_coeff_field=2, min_persistence=0)
    result = []
    for dim in (0, 1):
        finite = [(float(b), float(d)) for k, (b, d) in pairs
                  if k == dim and math.isfinite(b) and math.isfinite(d) and d - b >= 0.02]
        result.append(np.asarray(finite, dtype=np.float64).reshape(-1, 2))
    return result[0], result[1]


def relative_distance(before: tuple[np.ndarray, np.ndarray],
                      after: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    ratios = []
    for old, new in zip(before, after):
        life = old[:, 1] - old[:, 0] if len(old) else np.empty(0)
        span = max(float(life.max()) if len(life) else 0.0, 1e-6)
        value = float(gudhi.bottleneck_distance(old, new)) / span
        ratios.append(value)
    return ratios[0], ratios[1], max(ratios)


def raw_crop(row: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    group = zarr.open_group(row["local_path"], mode="r")
    image = group["img"]
    depth, height, width = image.shape
    z = int(rng.integers(depth))
    y = int(rng.integers(max(1, height - 512 + 1)))
    x = int(rng.integers(max(1, width - 512 + 1)))
    crop = np.asarray(image[z, y:y + 512, x:x + 512], dtype=np.float32) / 255.0
    py, px = 512 - crop.shape[0], 512 - crop.shape[1]
    if py or px:
        crop = np.pad(crop, ((0, py), (0, px)),
                      mode='reflect' if min(crop.shape) > 1 else 'edge')
    return crop, {"volume_id": row["volume_id"], "z": z, "y": y, "x": x,
                  "valid_height": min(height - y, 512), "valid_width": min(width - x, 512)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260926)
    args = parser.parse_args()
    if not 1 <= args.per_source <= 64:
        raise SystemExit("per-source must be in 1..64")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    rows = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    audit = validate_manifest(args.train_manifest, read_center=False)
    if not rows or any(r.get('training_role') != 'ssl_full_mitoverse' for r in rows):
        raise SystemExit('Calibration requires the full MitoVerse SSL manifest')
    groups = defaultdict(list)
    for row in rows:
        groups[row["original_dataset_id"]].append(row)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    rng = np.random.default_rng(args.seed)
    detail = []
    first_crop = None
    for source in sorted(groups):
        for sample_idx in range(args.per_source):
            row = groups[source][int(rng.integers(len(groups[source])))]
            raw, location = raw_crop(row, rng)
            if first_crop is None:
                first_crop = raw
            orig64 = downsample(raw)
            original_diagrams = diagrams(orig64)
            for candidate_idx, candidate in enumerate(CANDIDATES):
                aug = appearance(raw, candidate, args.seed + len(detail) * 17 + candidate_idx)
                aug64 = downsample(aug)
                h0, h1, combined = relative_distance(original_diagrams, diagrams(aug64))
                detail.append({"source": source, **location, "sample_index": sample_idx,
                               "candidate": candidate["name"], "h0_relative": h0,
                               "h1_relative": h1, "relative_max": combined,
                               "appearance_mae": float(np.abs(aug64 - orig64).mean())})
            if sample_idx % 8 == 7:
                print(json.dumps({"source": source, "samples_completed": sample_idx + 1}), flush=True)
    write_csv(output / "topology_calibration_detail.csv", detail)
    summary = []
    for candidate in CANDIDATES:
        source_means = []
        appearance_means = []
        for source in sorted(groups):
            values = [r for r in detail if r["source"] == source and r["candidate"] == candidate["name"]]
            source_means.append(float(np.mean([r["relative_max"] for r in values])))
            appearance_means.append(float(np.mean([r["appearance_mae"] for r in values])))
        summary.append({"candidate": candidate["name"], "source_equal_relative_max": float(np.mean(source_means)),
                        "source_equal_appearance_mae": float(np.mean(appearance_means)),
                        "source_means": json.dumps(dict(zip(sorted(groups), source_means))),
                        "parameters": json.dumps(candidate, sort_keys=True)})
    summary.sort(key=lambda r: (r["source_equal_relative_max"], r["candidate"]))
    write_csv(output / "topology_calibration.csv", summary)
    scores = np.asarray([r["source_equal_relative_max"] for r in summary])
    if not np.isfinite(scores).all() or float(scores.max() - scores.min()) < 1e-4:
        decision = {"status": "blocked", "reason": "Candidate topology distances are nonfinite or indistinguishable"}
    else:
        width = max(2, len(summary) // 3)
        decision = {"status": "complete", "weak": [r["candidate"] for r in summary[:width]],
                    "strong": [r["candidate"] for r in summary[width:2 * width]],
                    "per_source": args.per_source, "seed": args.seed,
                    "manifest_sha256": audit['manifest_sha256'],
                    "source_count": audit['source_count'],
                    "volume_count": audit['volume_count'],
                    "filtration": "GUDHI CubicalComplex top-dimensional cells, lower star",
                    "resize": "skimage resize 64x64 order=1 reflect anti_aliasing=True",
                    "minimum_lifetime": 0.02,
                    "distance": "GUDHI bottleneck, H0/H1 divided by original max lifetime or 1e-6; take max",
                    "gudhi_version": gudhi.__version__}
        Image.fromarray(np.round(first_crop * 255).astype(np.uint8)).resize((256, 256)).save(output / "sample_original.png")
        for label in ("weak", "strong"):
            candidate = next(c for c in CANDIDATES if c["name"] == decision[label][0])
            image = appearance(first_crop, candidate, args.seed + (1 if label == "weak" else 2))
            Image.fromarray(np.round(image * 255).astype(np.uint8)).resize((256, 256)).save(output / f"sample_{label}.png")
    (output / "topology_selected.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision), flush=True)
    return 0 if decision["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
