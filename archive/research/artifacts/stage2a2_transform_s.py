"""Verify offline flip/rotation invariance of directional cut and H0 graph scores."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "src"))
from score_s import N, box_on_grid, cut_score, graphs, persistent_cut  # noqa: E402

REF = json.loads((RUN / "references/paths.json").read_text())
STAGE2 = Path(REF["stage2_root"])
MAPS = RUN / "branch_S/directional_affinity/maps"
OUT = RUN / "branch_S/graph_cut"
OPS = ("hflip", "vflip", "rot90", "rot180", "rot270")


def location(r: int, c: int, op: str) -> tuple[int, int]:
    if op == "hflip": return r, N - 1 - c
    if op == "vflip": return N - 1 - r, c
    if op == "rot90": return N - 1 - c, r
    if op == "rot180": return N - 1 - r, N - 1 - c
    if op == "rot270": return c, N - 1 - r
    raise ValueError(op)


def transform(ay: np.ndarray, ax: np.ndarray, op: str) -> tuple[np.ndarray, np.ndarray]:
    ty, tx = np.zeros_like(ay), np.zeros_like(ax)
    for r in range(N):
        for c in range(N):
            if r > 0:
                a, b = location(r - 1, c, op), location(r, c, op)
                if a[0] != b[0]: ty[max(a[0], b[0]), a[1]] = ay[r, c]
                else: tx[a[0], max(a[1], b[1])] = ay[r, c]
            if c > 0:
                a, b = location(r, c - 1, op), location(r, c, op)
                if a[0] != b[0]: ty[max(a[0], b[0]), a[1]] = ax[r, c]
                else: tx[a[0], max(a[1], b[1])] = ax[r, c]
    assert np.array_equal(np.sort(np.concatenate((ay[1:].ravel(), ax[:, 1:].ravel()))),
                          np.sort(np.concatenate((ty[1:].ravel(), tx[:, 1:].ravel()))))
    return ty, tx


def transform_box(box: tuple[int, int, int, int], op: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    corners = [location(r, c, op) for r in (r0, r1 - 1) for c in (c0, c1 - 1)]
    return min(r for r, _ in corners), min(c for _, c in corners), max(r for r, _ in corners) + 1, max(c for _, c in corners) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=424)
    ap.add_argument("--graph-limit", type=int, default=25)
    args = ap.parse_args()
    with (STAGE2 / "topology_calibration/crop_score/crop_score_pilot_424x4.csv").open(newline="") as f:
        candidates = list(csv.DictReader(f))
    assert len(candidates) == 1696
    metrics = ("cut_mean", "cut_top25", "cut_fraction_q75", "cut_fraction_q90")
    diffs = {name: [] for name in metrics + ("persist_include_largest", "persist_exclude_largest")}
    for index in range(args.limit):
        with np.load(MAPS / f"sample_{index:06d}.npz") as pack:
            ay, ax = pack["ay"].copy(), pack["ax"].copy()
        boxes = [box_on_grid(r) for r in candidates[index * 4:(index + 1) * 4]]
        original = [cut_score(ay, ax, box) for box in boxes]
        if index < args.graph_limit:
            old_graph, _ = graphs(ay, ax)
            old_persist = [persistent_cut(old_graph, box) for box in boxes]
        for op in OPS:
            ty, tx = transform(ay, ax, op)
            tboxes = [transform_box(box, op) for box in boxes]
            for old, box in zip(original, tboxes):
                new = cut_score(ty, tx, box)
                for name in metrics:
                    if np.isfinite(old[name]) and np.isfinite(new[name]):
                        diffs[name].append(abs(old[name] - new[name]))
            if index < args.graph_limit:
                new_graph, _ = graphs(ty, tx)
                for old, box in zip(old_persist, tboxes):
                    new = persistent_cut(new_graph, box)
                    diffs["persist_include_largest"].append(abs(old[0] - new[0]))
                    diffs["persist_exclude_largest"].append(abs(old[1] - new[1]))
        if index % 25 == 0 or index + 1 == args.limit:
            print(f"transformed {index + 1}/{args.limit}", flush=True)
    result = {"status": "complete", "samples": args.limit, "graph_samples": min(args.limit, args.graph_limit),
              "transforms": OPS, "metrics": {name: {"comparisons": len(vals), "max_abs_difference": max(vals) if vals else None,
                                              "exact_fraction": float(np.mean(np.asarray(vals) == 0)) if vals else None}
                                          for name, vals in diffs.items()}}
    (OUT / f"transform_invariance_{args.limit:03d}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
