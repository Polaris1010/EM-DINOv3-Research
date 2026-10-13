"""Same-shape crop-location pilot with directional cut and H0 graph connectivity."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

RUN = Path(__file__).resolve().parents[1]
REF = json.loads((RUN / "references/paths.json").read_text())
STAGE2 = Path(REF["stage2_root"])
MAPS = RUN / "branch_S/directional_affinity/maps"
OUT = RUN / "branch_S/graph_cut"
N = 64
QUANTILES = (.50, .60, .70, .80, .90)


def box_on_grid(row: dict) -> tuple[int, int, int, int]:
    y, x, h, w = (int(row[k]) for k in ("y", "x", "h", "w"))
    r0 = min(N - 1, int(math.floor(y * N / 512)))
    c0 = min(N - 1, int(math.floor(x * N / 512)))
    r1 = min(N, max(r0 + 1, int(math.ceil((y + h) * N / 512))))
    c1 = min(N, max(c0 + 1, int(math.ceil((x + w) * N / 512))))
    return r0, c0, r1, c1


def parent(box: tuple[int, int, int, int], eta: float = 1.5) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    h, w = r1 - r0, c1 - c0
    ph, pw = min(N, round(h * eta)), min(N, round(w * eta))
    cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
    pr0 = max(0, min(N - ph, round(cr - ph / 2)))
    pc0 = max(0, min(N - pw, round(cc - pw / 2)))
    return pr0, pc0, pr0 + ph, pc0 + pw


def internal_edges(ay: np.ndarray, ax: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.concatenate((ay[r0 + 1:r1, c0:c1].ravel(), ax[r0:r1, c0 + 1:c1].ravel()))


def crossing_edges(ay: np.ndarray, ax: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, float, dict]:
    r0, c0, r1, c1 = box
    sides = {
        "top": ay[r0, c0:c1] if r0 > 0 else np.empty(0, np.float32),
        "bottom": ay[r1, c0:c1] if r1 < N else np.empty(0, np.float32),
        "left": ax[r0:r1, c0] if c0 > 0 else np.empty(0, np.float32),
        "right": ax[r0:r1, c1] if c1 < N else np.empty(0, np.float32),
    }
    edges = np.concatenate(list(sides.values()))
    expected = 2 * ((r1 - r0) + (c1 - c0))
    return edges, len(edges) / expected, {k: len(v) for k, v in sides.items()}


def cut_score(ay: np.ndarray, ax: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    cross, coverage, side_counts = crossing_edges(ay, ax, box)
    context = internal_edges(ay, ax, parent(box))
    assert len(context)
    q75, q90 = np.quantile(context, [.75, .90])
    if not len(cross):
        return {"cut_mean": float("nan"), "cut_top25": float("nan"),
                "cut_fraction_q75": float("nan"), "cut_fraction_q90": float("nan"),
                "cross_edges": 0, "boundary_coverage": 0.0,
                "all_four_sides_observed": False, "context_q75": float(q75), "context_q90": float(q90)}
    n_top = max(1, math.ceil(.25 * len(cross)))
    return {"cut_mean": float(np.mean(cross)),
            "cut_top25": float(np.mean(np.partition(cross, -n_top)[-n_top:])),
            "cut_fraction_q75": float(np.mean(cross > q75)),
            "cut_fraction_q90": float(np.mean(cross > q90)),
            "cross_edges": len(cross), "boundary_coverage": coverage,
            "all_four_sides_observed": all(side_counts.values()),
            "context_q75": float(q75), "context_q90": float(q90)}


def graphs(ay: np.ndarray, ax: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], dict]:
    ids = np.arange(N * N).reshape(N, N)
    u = np.concatenate((ids[1:, :].ravel(), ids[:, 1:].ravel()))
    v = np.concatenate((ids[:-1, :].ravel(), ids[:, :-1].ravel()))
    weights = np.concatenate((ay[1:, :].ravel(), ax[:, 1:].ravel()))
    thresholds = np.quantile(weights, QUANTILES)
    result = []
    largest_fractions = []
    for threshold in thresholds:
        selected = weights >= threshold
        uu, vv = u[selected], v[selected]
        graph = coo_matrix((np.ones(2 * len(uu), dtype=np.uint8),
                            (np.concatenate((uu, vv)), np.concatenate((vv, uu)))),
                           shape=(N * N, N * N)).tocsr()
        count, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels, minlength=count)
        largest = int(np.argmax(sizes))
        largest_fractions.append(float(sizes[largest] / (N * N)))
        result.append((labels.reshape(N, N), sizes, largest))
    return result, {"thresholds": [float(x) for x in thresholds],
                    "largest_component_fraction": largest_fractions}


def persistent_cut(graph_data: list[tuple[np.ndarray, np.ndarray, int]], box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    include = exclude = 0
    for labels, sizes, largest in graph_data:
        counts = np.bincount(labels[r0:r1, c0:c1].ravel(), minlength=len(sizes))
        crosses = (counts > 0) & (counts < sizes)
        include += int(np.count_nonzero(crosses))
        # A tie has no rotation-invariant single "largest" component. Keep all
        # tied components; exclude only a unique giant component.
        if np.count_nonzero(sizes == sizes[largest]) == 1:
            crosses[largest] = False
        exclude += int(np.count_nonzero(crosses))
    return include / len(graph_data), exclude / len(graph_data)


def rank_concordance(a: list[float], b: list[float]) -> float:
    valid = equal = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if not (np.isfinite(da) and np.isfinite(db)):
                continue
            if abs(da) < 1e-8:
                continue
            valid += 1
            equal += (da * db > 0)
    return float(equal / valid) if valid else float("nan")


def describe(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "median": None, "p25": None, "p75": None, "p95": None}
    return {"n": len(a), "median": float(np.median(a)), "p25": float(np.quantile(a, .25)),
            "p75": float(np.quantile(a, .75)), "p95": float(np.quantile(a, .95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=424)
    args = ap.parse_args()
    assert 1 <= args.limit <= 424
    samples = json.loads((RUN / "references/sample_manifest_424.json").read_text())["samples"]
    assert len(samples) == 424
    with (STAGE2 / "topology_calibration/crop_score/crop_score_pilot_424x4.csv").open(newline="") as f:
        candidates = list(csv.DictReader(f))
    assert len(candidates) == 1696
    rows = []
    sample_summary = []
    score_names = ("cut_mean", "cut_top25", "cut_fraction_q75", "cut_fraction_q90", "persist_include_largest", "persist_exclude_largest")
    stability = defaultdict(list)
    for index, sample in enumerate(samples[:args.limit]):
        with np.load(MAPS / f"sample_{index:06d}.npz") as pack:
            ay, ax = pack["ay"].astype(np.float32), pack["ax"].astype(np.float32)
        group = candidates[index * 4:(index + 1) * 4]
        assert all(int(r["sample_index"]) == index for r in group)
        graph_data, graph_meta = graphs(ay, ax)
        boxes = [box_on_grid(r) for r in group]
        scored = []
        for old, box in zip(group, boxes):
            cut = cut_score(ay, ax, box)
            include, exclude = persistent_cut(graph_data, box)
            scored.append({"sample_index": index, "volume_id": sample["volume_id"], "source": sample["source"],
                           "candidate": int(old["candidate"]), "old_pd_score": float(old["topo_score"]),
                           "grid_r0": box[0], "grid_c0": box[1], "grid_r1": box[2], "grid_c1": box[3],
                           **cut, "persist_include_largest": include, "persist_exclude_largest": exclude})
        for dtype_name, alt_ay, alt_ax in (("float16", ay.astype(np.float16).astype(np.float32), ax.astype(np.float16).astype(np.float32)),
                                           ("jitter_001", np.clip(ay + np.random.default_rng(index).normal(0, .001, ay.shape), 0, 1).astype(np.float32),
                                            np.clip(ax + np.random.default_rng(index + 424).normal(0, .001, ax.shape), 0, 1).astype(np.float32))):
            alt_graphs, _ = graphs(alt_ay, alt_ax)
            alt = []
            for box in boxes:
                c = cut_score(alt_ay, alt_ax, box)
                inc, exc = persistent_cut(alt_graphs, box)
                alt.append({**c, "persist_include_largest": inc, "persist_exclude_largest": exc})
            for name in score_names:
                stability[f"{name}:{dtype_name}"].append(rank_concordance([r[name] for r in scored], [r[name] for r in alt]))
        def finite_range(name: str) -> float:
            vals = np.asarray([r[name] for r in scored], dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(np.ptp(vals)) if len(vals) >= 2 else float("nan")
        sample_summary.append({"sample_index": index, "source": sample["source"],
                               "largest_component_fraction_q50": graph_meta["largest_component_fraction"][0],
                               "largest_component_fraction_q90": graph_meta["largest_component_fraction"][-1],
                               **{f"range_{name}": finite_range(name) for name in score_names}})
        rows.extend(scored)
        if index % 25 == 0 or index + 1 == args.limit:
            print(f"scored {index + 1}/{args.limit}", flush=True)
    suffix = f"{args.limit:03d}"
    with (OUT / f"same_shape_scores_{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / f"sample_summary_{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_summary[0]))
        writer.writeheader()
        writer.writerows(sample_summary)
    summary = {"status": "complete", "samples": args.limit, "candidates": len(rows),
               "prior": "frozen SAM ViT-H native 64x64 directional affinity",
               "candidate_source": "Stage2 424x4 matched-shape sampled positions",
               "thresholds": "per-sample global directional affinity Q50,Q60,Q70,Q80,Q90",
               "persistent_weight": "one per crossing connected component per threshold",
               "score_median_range": {name: describe([r[f"range_{name}"] for r in sample_summary]) for name in score_names},
               "nonzero_range_fraction": {name: float(np.mean([r[f"range_{name}"] > 1e-6 for r in sample_summary])) for name in score_names},
               "ranking_stability": {key: describe(value) for key, value in stability.items()},
               "full_boundary_observed_fraction": float(np.mean([r["all_four_sides_observed"] for r in rows])),
               "largest_component_fraction_q50": describe([r["largest_component_fraction_q50"] for r in sample_summary]),
               "largest_component_fraction_q90": describe([r["largest_component_fraction_q90"] for r in sample_summary]),
               "by_source": {src: {name: describe([r[f"range_{name}"] for r in sample_summary if r["source"] == src]) for name in score_names}
                             for src in sorted({r["source"] for r in sample_summary})}}
    (OUT / f"summary_{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_source"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
