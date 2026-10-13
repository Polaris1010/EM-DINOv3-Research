"""Validate paired Stage 2A slices and audit Full MitoVerse image geometry."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: np.ndarray) -> dict:
    return {"min": float(values.min()), "p25": float(np.quantile(values, .25)), "median": float(np.median(values)), "p75": float(np.quantile(values, .75)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def tiles_for_axis(size: int, tile: int = 1024, stride: int = 768) -> int:
    return 1 if size <= tile else math.ceil((size - tile) / stride) + 1


def main() -> None:
    paths = json.loads((RUN / "stage1_reference/stage1_paths.json").read_text())
    stage1 = Path(paths["stage1_root"])
    full_path = Path(paths["manifest"])
    sample_path = RUN / "topology_calibration/sample_manifest.json"
    assert sha256(full_path) == (RUN / "stage1_reference/manifest.sha256").read_text().strip()
    assert sha256(sample_path) == sha256(stage1 / "topology_feasibility/sample_manifest.json")
    full = json.loads(full_path.read_text())
    sample = json.loads(sample_path.read_text())
    volumes = full["volumes"]
    samples = sample["samples"]
    assert len(volumes) == 212 and len(samples) == 424
    volume_by_id = {v["volume_id"]: v for v in volumes}
    assert len(volume_by_id) == len(volumes)
    counts = Counter()
    for s in samples:
        v = volume_by_id[s["volume_id"]]
        counts[s["volume_id"]] += 1
        z, h, w = map(int, v["shape"])
        assert 0 <= int(s["slice_index"]) < z
        assert s["source"] == v["source_dataset"]
        assert list(s["image_shape"]) == [h, w]
        assert s["local_path"] == v["local_path"]
        assert s["image_array_path"] == v["image_array_path"]
    assert set(counts) == set(volume_by_id) and set(counts.values()) == {2}
    report = {"status": "pass", "sample_count": len(samples), "volume_count": len(volumes), "samples_per_volume": 2, "source_count": len({s['source'] for s in samples}), "stage1_sample_manifest_sha256": sha256(sample_path), "full_manifest_sha256": sha256(full_path), "label_access": False}
    (RUN / "audit/sample_validation.json").write_text(json.dumps(report, indent=2))

    by_source = defaultdict(list)
    rows = []
    for v in volumes:
        z, h, w = map(int, v["shape"])
        row = {"volume_id": v["volume_id"], "source": v["source_dataset"], "slices": z, "height": h, "width": w, "aspect_hw": h / w, "pixels_per_slice": h * w, "tiles_1024_overlap25_per_slice": tiles_for_axis(h) * tiles_for_axis(w)}
        rows.append(row)
        by_source[row["source"]].append(row)
    geometry_path = RUN / "topology_calibration/offline_prior/slice_geometry_by_volume.csv"
    with geometry_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats = {}
    for source, members in [("ALL", rows), *sorted(by_source.items())]:
        stats[source] = {"volumes": len(members), "slices": sum(r["slices"] for r in members), "weighted_by_slice": {}}
        for key in ("height", "width", "aspect_hw", "pixels_per_slice"):
            values = np.repeat(np.asarray([r[key] for r in members], dtype=np.float64), [r["slices"] for r in members])
            stats[source]["weighted_by_slice"][key] = describe(values)
    total_slices = sum(r["slices"] for r in rows)
    tiles_total = sum(r["slices"] * r["tiles_1024_overlap25_per_slice"] for r in rows)
    assert total_slices == 48722
    result = {"status": "complete", "slice_count": total_slices, "volume_count": len(rows), "full_slice_max_dim_gt_1024": sum(r["slices"] for r in rows if max(r["height"], r["width"]) > 1024), "total_pixels": sum(r["slices"] * r["pixels_per_slice"] for r in rows), "estimated_1024_overlap25_tiles": tiles_total, "estimated_native_mam_float16_bytes_at_64x64_per_tile": tiles_total * 64 * 64 * 2, "stats": stats}
    (RUN / "topology_calibration/offline_prior/slice_geometry_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "stats"}, indent=2))
    print('all_weighted_slice_stats', json.dumps(stats['ALL']['weighted_by_slice']))


if __name__ == "__main__":
    main()
