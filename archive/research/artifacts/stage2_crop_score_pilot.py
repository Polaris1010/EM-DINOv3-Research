"""Label-free Stage 2A crop/context topology score pilot on fixed ViT-H priors."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gudhi
import numpy as np
from scipy.stats import spearmanr

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "src"))
from equivariance_b import diagrams  # noqa: E402

MAPS = RUN / "topology_calibration/sam_vit_h/paired_maps"
OUT = RUN / "topology_calibration/crop_score"


def rrc_shape(rng: np.random.Generator, side: int = 512) -> tuple[int, int]:
    area = side * side
    for _ in range(10):
        target = area * rng.uniform(.32, 1.0)
        aspect = math.exp(rng.uniform(math.log(.75), math.log(4 / 3)))
        w = round(math.sqrt(target * aspect))
        h = round(math.sqrt(target / aspect))
        if 0 < w <= side and 0 < h <= side:
            return h, w
    return side, side


def box_on_prior(box: tuple[int, int, int, int], side: int = 512, grid: int = 64) -> tuple[int, int, int, int]:
    y, x, h, w = box
    y0 = min(grid - 1, int(math.floor(y * grid / side)))
    x0 = min(grid - 1, int(math.floor(x * grid / side)))
    y1 = min(grid, max(y0 + 1, int(math.ceil((y + h) * grid / side))))
    x1 = min(grid, max(x0 + 1, int(math.ceil((x + w) * grid / side))))
    return y0, x0, y1, x1


def parent(box: tuple[int, int, int, int], side: int = 512, eta: float = 1.5) -> tuple[int, int, int, int]:
    y, x, h, w = box
    ph, pw = min(side, round(h * eta)), min(side, round(w * eta))
    cy, cx = y + h / 2, x + w / 2
    py = max(0, min(side - ph, round(cy - ph / 2)))
    px = max(0, min(side - pw, round(cx - pw / 2)))
    return py, px, ph, pw


def extract(mam: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, y1, x1 = box_on_prior(box)
    return np.ascontiguousarray(mam[y0:y1, x0:x1])


def score(mam: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float, float]:
    crop = extract(mam, box)
    context = extract(mam, parent(box))
    d_c = diagrams(crop)
    d_p = diagrams(context)
    d0 = float(gudhi.bottleneck_distance(d_c[0], d_p[0], e=0))
    d1 = float(gudhi.bottleneck_distance(d_c[1], d_p[1], e=0))
    boundary = 1.0 - crop
    band = max(1, round(min(crop.shape) * .05))
    border = np.zeros(crop.shape, dtype=bool)
    border[:band] = border[-band:] = True
    border[:, :band] = border[:, -band:] = True
    border_proxy = float(boundary[border].mean() / max(boundary.mean(), 1e-6))
    return d0, d1, border_proxy


def robust(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(np.median(a)), float(np.quantile(a, .75) - np.quantile(a, .25) + 1e-6)


def main() -> None:
    samples = json.loads((RUN / "topology_calibration/sample_manifest.json").read_text())["samples"]
    assert len(samples) == 424
    assert json.loads((RUN / "topology_calibration/sam_vit_h/generation_complete.json").read_text())["status"] == "complete"
    rng = np.random.default_rng(20260930)
    rows = []
    for index, sample in enumerate(samples):
        with np.load(MAPS / f"sample_{index:06d}.npz") as pack:
            mam = pack["mam"][0]
        h, w = rrc_shape(rng)
        for candidate in range(4):
            y = int(rng.integers(513 - h))
            x = int(rng.integers(513 - w))
            d0, d1, proxy = score(mam, (y, x, h, w))
            rows.append({"sample_index": index, "volume_id": sample["volume_id"], "source": sample["source"],
                         "candidate": candidate, "y": y, "x": x, "h": h, "w": w,
                         "area_fraction": h * w / 512**2, "aspect": w / h,
                         "d0": d0, "d1": d1, "border_proxy": proxy})
        if index % 50 == 0:
            print(f"scored {index + 1}/424", flush=True)
    areas = np.array([r["area_fraction"] for r in rows])
    q1, q2 = np.quantile(areas, [1 / 3, 2 / 3])
    source_volumes = defaultdict(set)
    for row in rows:
        source_volumes[row["source"]].add(row["volume_id"])
        row["scale_bin"] = int(row["area_fraction"] > q1) + int(row["area_fraction"] > q2)
    groups = defaultdict(list)
    for row in rows:
        groups[("GLOBAL", row["scale_bin"])].append(row)
        if len(source_volumes[row["source"]]) >= 3:
            groups[(row["source"], row["scale_bin"])].append(row)
    normalizers = {}
    for key, group in groups.items():
        normalizers[f"{key[0]}:{key[1]}"] = {"n": len(group), "independent_volumes": len({r["volume_id"] for r in group}),
              "d0": robust([r["d0"] for r in group]), "d1": robust([r["d1"] for r in group])}
    for row in rows:
        source = row["source"] if len(source_volumes[row["source"]]) >= 3 else "GLOBAL"
        norm = normalizers[f"{source}:{row['scale_bin']}"]
        row["normalizer_source"] = source
        row["topo_score"] = .5 * (row["d0"] - norm["d0"][0]) / norm["d0"][1] + .5 * (row["d1"] - norm["d1"][0]) / norm["d1"][1]
    scores = np.array([r["topo_score"] for r in rows])
    proxies = np.array([r["border_proxy"] for r in rows])
    thresholds = {f"Q{q}": float(np.quantile(scores, q / 100)) for q in (85, 90, 95)}
    policies = {}
    for label, threshold in thresholds.items():
        chosen = []
        fallbacks = 0
        for index in range(424):
            candidates = rows[index * 4:(index + 1) * 4]
            valid = [r for r in candidates if r["topo_score"] <= threshold]
            if valid:
                choice = valid[int(rng.integers(len(valid)))]
            else:
                choice = min(candidates, key=lambda r: r["topo_score"])
                fallbacks += 1
            chosen.append(choice)
        policies[label] = {"fallback_rate": fallbacks / 424,
                           "mean_border_proxy": float(np.mean([r["border_proxy"] for r in chosen])),
                           "mean_score": float(np.mean([r["topo_score"] for r in chosen]))}
    rho, pvalue = spearmanr(scores, proxies)
    result = {"status": "complete", "sample_count": 424, "candidate_count": len(rows),
              "prior": "frozen SAM ViT-H, centered Stage1-normalized 512 patch, 64x64 MAM",
              "global_crop_scale": [.32, 1.0], "context_expansion": 1.5,
              "area_bin_edges": [float(q1), float(q2)],
              "d0_quantiles": [float(x) for x in np.quantile([r["d0"] for r in rows], [.05, .25, .5, .75, .95])],
              "d1_quantiles": [float(x) for x in np.quantile([r["d1"] for r in rows], [.05, .25, .5, .75, .95])],
              "score_quantiles": [float(x) for x in np.quantile(scores, [.05, .25, .5, .75, .95])],
              "score_border_proxy_spearman": float(rho), "score_border_proxy_pvalue": float(pvalue),
              "random_mean_border_proxy": float(proxies.mean()), "thresholds": thresholds,
              "policies": policies,
              "small_source_global_fallbacks": sorted([s for s, ids in source_volumes.items() if len(ids) < 3]),
              "normalizers": normalizers}
    with (OUT / "crop_score_pilot_424x4.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "crop_score_pilot_424x4.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "normalizers"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
