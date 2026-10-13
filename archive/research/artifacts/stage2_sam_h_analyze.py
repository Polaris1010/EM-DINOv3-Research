"""Paired ViT-B/ViT-H prior statistics and ViT-H equivariance audit."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "src"))
from equivariance_b import OPS, diagrams, metrics, quantiles  # noqa: E402

STAGE1 = Path(json.loads((RUN / "stage1_reference/stage1_paths.json").read_text())["stage1_root"])
B_MAPS = STAGE1 / "topology_feasibility/mam_sam_vit_b/paired/maps"
H_MAPS = RUN / "topology_calibration/sam_vit_h/paired_maps"
OUT = RUN / "topology_calibration"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def map_features(x: np.ndarray) -> dict:
    hist, _ = np.histogram(x, bins=32, range=(0, 1))
    p = hist.astype(np.float64) / x.size
    p = p[p > 0]
    d0, d1 = diagrams(x)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()), "dynamic_range": float(x.max() - x.min()),
            "entropy_32": float(-(p * np.log2(p)).sum()),
            "edge_density_005": float(((np.abs(np.diff(x, axis=0)) > .05).mean() + (np.abs(np.diff(x, axis=1)) > .05).mean()) / 2),
            "neighbor_autocorr": float((corr(x[1:, :], x[:-1, :]) + corr(x[:, 1:], x[:, :-1])) / 2),
            "h0_count": len(d0), "h1_count": len(d1),
            "h0_mass": float(np.maximum(d0[:, 1] - d0[:, 0], 0).sum()),
            "h1_mass": float(np.maximum(d1[:, 1] - d1[:, 0], 0).sum())}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    assert json.loads((RUN / "audit/sample_validation.json").read_text())["status"] == "pass"
    assert json.loads((OUT / "sam_vit_h/generation_complete.json").read_text())["status"] == "complete"
    samples = json.loads((OUT / "sample_manifest.json").read_text())["samples"]
    assert len(samples) == 424
    paired, equivariance = [], []
    for index, sample in enumerate(samples):
        with np.load(B_MAPS / f"sample_{index:06d}.npz") as b_data:
            b = b_data["mam"][0]
        with np.load(H_MAPS / f"sample_{index:06d}.npz") as h_data:
            h = h_data["mam"]
        assert b.shape == (64, 64) and h.shape == (6, 64, 64)
        for name, x in (("ViT-B", b), ("ViT-H", h[0])):
            paired.append({"sample_index": index, "volume_id": sample["volume_id"],
                           "source": sample["source"], "model": name, **map_features(x)})
        pd0 = diagrams(h[0])
        for variant_index, op in enumerate(OPS, 1):
            inverse = OPS[op][1]
            aligned = inverse(h[variant_index]).copy()
            equivariance.append({"sample_index": index, "volume_id": sample["volume_id"],
                                "source": sample["source"], "path": "online_H", "transform": op,
                                **metrics(h[0], aligned, pd0)})
            offline = inverse(OPS[op][0](h[0])).copy()
            equivariance.append({"sample_index": index, "volume_id": sample["volume_id"],
                                "source": sample["source"], "path": "offline_H", "transform": op,
                                **metrics(h[0], offline, pd0)})
        if index % 25 == 0:
            print(f"analyzed {index + 1}/424", flush=True)
    write_csv(OUT / "sam_vit_h/paired_B_H_features_424.csv", paired)
    write_csv(OUT / "equivariance/vit_h_equivariance_424.csv", equivariance)
    features = list(map_features(np.zeros((64, 64), dtype=np.float32)))
    feature_summary = {}
    for model in ("ViT-B", "ViT-H"):
        rows = [r for r in paired if r["model"] == model]
        feature_summary[model] = {key: {"median": float(np.median([r[key] for r in rows])),
                                        "p95": float(np.quantile([r[key] for r in rows], .95))} for key in features}
        feature_summary[model]["by_source"] = {source: {"n": len(group), "mean_mam_std": float(np.mean([r["std"] for r in group]))}
                for source in sorted({r["source"] for r in rows})
                if (group := [r for r in rows if r["source"] == source])}
    (OUT / "sam_vit_h/paired_B_H_summary.json").write_text(json.dumps(feature_summary, indent=2))
    groups = {path: {op: quantiles([r for r in equivariance if r["path"] == path and r["transform"] == op])
                     for op in OPS} for path in ("online_H", "offline_H")}
    offline = [r for r in equivariance if r["path"] == "offline_H"]
    gate = all(r["equal_bits"] and r["max_abs_diff"] == 0 and r["d0"] == 0 and r["d1"] == 0 for r in offline)
    result = {"status": "pass" if gate else "fail", "sample_count": 424, "groups": groups,
              "offline_H_exact_count": sum(r["equal_bits"] for r in offline), "offline_H_total": len(offline)}
    (OUT / "equivariance/vit_h_equivariance_424.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "online_H_hflip": groups["online_H"]["hflip"],
                      "offline_H_exact_count": result["offline_H_exact_count"]}, indent=2))
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
