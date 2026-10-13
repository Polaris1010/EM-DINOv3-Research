"""Exclude near-identity calibrated candidates before locking weak/strong lists."""

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-appearance-mae", type=float, default=0.001)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.summary.open(newline="", encoding="utf-8")))
    eligible = [r for r in rows if float(r["source_equal_appearance_mae"]) >= args.minimum_appearance_mae]
    excluded = [{"candidate": r["candidate"],
                 "source_equal_appearance_mae": float(r["source_equal_appearance_mae"])}
                for r in rows if r not in eligible]
    eligible.sort(key=lambda r: (float(r["source_equal_relative_max"]), r["candidate"]))
    width = max(2, len(eligible) // 3)
    if len(eligible) < 6 or len(eligible[width:2 * width]) < 2:
        result = {"status": "blocked", "reason": "Fewer than two eligible weak or strong candidates"}
    else:
        result = {
            "status": "complete",
            "weak": [r["candidate"] for r in eligible[:width]],
            "strong": [r["candidate"] for r in eligible[width:2 * width]],
            "excluded_near_identity": excluded,
            "minimum_source_equal_appearance_mae_at_64x64": args.minimum_appearance_mae,
            "selection_rule": "Among non-degenerate candidates, lower third weak, middle third strong; source-equal PH distance",
            "decision_note": "Technical exclusion before any SSL model training; original calibration scores retained",
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
