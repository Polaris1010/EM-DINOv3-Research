"""Stage 2A ViT-B offline geometry hard gate and online flip diagnosis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import gudhi
import numpy as np
from skimage.metrics import structural_similarity


RUN = Path(__file__).resolve().parents[1]
STAGE1 = Path(json.loads((RUN / "stage1_reference/stage1_paths.json").read_text())["stage1_root"])
MAPS = STAGE1 / "topology_feasibility/mam_sam_vit_b/paired/maps"
OUT = RUN / "topology_calibration/equivariance"
OPS = {
    "hflip": (lambda x: np.flip(x, 1), lambda x: np.flip(x, 1)),
    "vflip": (lambda x: np.flip(x, 0), lambda x: np.flip(x, 0)),
    "rot90": (lambda x: np.rot90(x, 1), lambda x: np.rot90(x, -1)),
    "rot180": (lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 2)),
    "rot270": (lambda x: np.rot90(x, 3), lambda x: np.rot90(x, 1)),
}


def diagrams(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cc = gudhi.CubicalComplex(top_dimensional_cells=np.ascontiguousarray(x).ravel(), dimensions=x.shape)
    cc.persistence()
    diagrams = []
    for dim in (0, 1):
        intervals = np.asarray(cc.persistence_intervals_in_dimension(dim), dtype=np.float64).reshape(-1, 2)
        intervals = intervals[np.isfinite(intervals).all(axis=1)]
        intervals = intervals[intervals[:, 1] - intervals[:, 0] >= .02]
        diagrams.append(intervals)
    return diagrams[0], diagrams[1]


def metrics(original: np.ndarray, candidate: np.ndarray, pd_original: tuple[np.ndarray, np.ndarray]) -> dict:
    a = original.astype(np.float64).ravel()
    b = candidate.astype(np.float64).ravel()
    diff = a - b
    pd_candidate = diagrams(candidate)
    result = {
        "mae": float(np.mean(np.abs(diff))), "mse": float(np.mean(diff * diff)),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else (1.0 if np.array_equal(a, b) else 0.0),
        "ssim": float(structural_similarity(original, candidate, data_range=1.0)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "equal_bits": bool(np.array_equal(original, candidate)),
        "d0": float(gudhi.bottleneck_distance(pd_original[0], pd_candidate[0], e=0)),
        "d1": float(gudhi.bottleneck_distance(pd_original[1], pd_candidate[1], e=0)),
    }
    return result


def quantiles(records: list[dict]) -> dict:
    return {key: {"median": float(np.median([r[key] for r in records])), "p95": float(np.quantile([r[key] for r in records], .95)), "max": float(max(r[key] for r in records))} for key in ("mae", "mse", "pearson_r", "ssim", "max_abs_diff", "d0", "d1")}


def main() -> None:
    assert json.loads((RUN / "audit/sample_validation.json").read_text())["status"] == "pass"
    samples = json.loads((RUN / "topology_calibration/sample_manifest.json").read_text())["samples"]
    assert len(samples) == 424
    rows = []
    for index, sample in enumerate(samples):
        with np.load(MAPS / f"sample_{index:06d}.npz") as package:
            mam = package["mam"]
        original = mam[0]
        assert original.shape == (64, 64) and original.dtype == np.float32
        pd_original = diagrams(original)
        for name, (forward, inverse) in OPS.items():
            restored = inverse(forward(original)).copy()
            rows.append({"sample_index": index, "volume_id": sample["volume_id"], "source": sample["source"], "path": "offline_B", "transform": name, **metrics(original, restored, pd_original)})
        for name, variant_index in (("hflip", 1), ("vflip", 2)):
            aligned = OPS[name][1](mam[variant_index]).copy()
            rows.append({"sample_index": index, "volume_id": sample["volume_id"], "source": sample["source"], "path": "online_B", "transform": name, **metrics(original, aligned, pd_original)})
    with (OUT / "vit_b_equivariance_424.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for path in ("offline_B", "online_B"):
        groups[path] = {}
        for transform in OPS if path == "offline_B" else ("hflip", "vflip"):
            group = [r for r in rows if r["path"] == path and r["transform"] == transform]
            assert len(group) == 424
            groups[path][transform] = quantiles(group)
    offline = [r for r in rows if r["path"] == "offline_B"]
    gate = all(r["equal_bits"] and r["max_abs_diff"] <= 1e-7 for r in offline)
    gate &= all(groups["offline_B"][op]["d0"]["median"] < 1e-4 and groups["offline_B"][op]["d1"]["median"] < 1e-4 and groups["offline_B"][op]["d0"]["p95"] < 1e-3 and groups["offline_B"][op]["d1"]["p95"] < 1e-3 for op in OPS)
    result = {"status": "pass" if gate else "fail", "sample_count": 424, "offline_transform_count": len(offline), "online_B_flip_count": 848, "PH_bottleneck_e": 0, "groups": groups, "diagnosis": "If offline exact and online differs, the measured flip anomaly is caused by SAM re-encoding / prior non-equivariance, not the inverse geometry or PH pipeline."}
    (OUT / "vit_b_equivariance_424.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "offline_hflip": groups["offline_B"]["hflip"], "online_B_hflip": groups["online_B"]["hflip"]}, indent=2))
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
