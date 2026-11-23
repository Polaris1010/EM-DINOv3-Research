"""Assert the matched 200-step smoke gate or verify the shared 30k handoff."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def handoff(root):
    receipt = read(root / "step000030000_full_state.READY")
    folder = root / "runs/Clean_C0_100k/checkpoints/step000030000"
    assert receipt == read(folder / "metadata.json")
    assert receipt["step"] == 30000 and receipt["model"] == "Clean_C0_100k"
    assert sha(folder / "full.pth") == receipt["full_sha256"]
    assert sha(folder / "teacher_backbone.pth") == receipt["teacher_sha256"]
    print(json.dumps({"handoff": "verified", **receipt}))


def verify(root):
    names = ("Clean_C0_100k", "Clean_C1_0k_30k")
    stats = {"status": "PASS", "matching_sample_steps": 200, "ranks": {}}
    for rank in range(3):
        runs = []
        for name in names:
            out = root / "smoke" / name
            assert read(out / f"result_rank{rank}.json")["step"] == 200
            rows = [json.loads(line) for line in (out / f"steps_rank{rank}.jsonl").read_text().splitlines()]
            assert [row["step"] for row in rows] == list(range(1, 201))
            assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in rows)
            assert all(all(math.isfinite(v) for v in row["loss_components"].values()) for row in rows)
            peak = max(row["gpu_peak_used_bytes"] for row in rows)
            margin = rows[-1]["gpu_total_bytes"] - peak
            assert margin >= 1.5 * 2**30, (name, rank, margin / 2**30)
            speed = statistics.median(row["step_seconds"] for row in rows[20:])
            row_stats = {"peak_used_gib": peak / 2**30, "margin_gib": margin / 2**30,
                         "median_step_seconds": speed, "estimated_100k_hours": speed * 100000 / 3600}
            if name == names[1]:
                assert all(row["barrier_backward_seen"] and math.isfinite(row["barrier_raw"]) for row in rows)
                row_stats["weighted_barrier_over_total_mean"] = statistics.mean(
                    row["barrier_weighted"] / row["loss"] for row in rows)
                row_stats["weighted_barrier_over_total_final"] = rows[-1]["barrier_weighted"] / rows[-1]["loss"]
                row_stats["lambda_final"] = rows[-1]["lambda"]
            stats["ranks"][f"{name}/gpu{rank+1}"] = row_stats
            runs.append(rows)
        assert [(r["sample_digest"], r["sample_chain"]) for r in runs[0]] == [
            (r["sample_digest"], r["sample_chain"]) for r in runs[1]], rank
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--handoff", action="store_true")
    args = parser.parse_args()
    (handoff if args.handoff else verify)(args.root)
