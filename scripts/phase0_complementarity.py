"""Phase 0: Error complementarity analysis for SARR feasibility.

Computes per-region and per-source complementarity between P0 and B100k
predictions using the existing MV-3 confusion matrix counts. Then estimates
the oracle upper-bound from per-region error pattern analysis.

This script runs locally (CPU only) on the existing CSV data.
A companion script (phase0_oracle_pixelwise.py) runs on the server to compute
exact pixel-level oracle Dice from saved prediction NPY files.

Usage:
    python scripts/phase0_complementarity.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments" / "MV-3" / "v2" / "MV3_per_region_metrics.csv"
OUT_DIR = ROOT / "experiments" / "SARR-Phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_mv3_metrics():
    """Load MV-3 per-region metrics for first seed only (seed 20261014)."""
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r["seed"]) == 20261014:  # first matched seed
                rows.append(r)
    return rows


def compute_complementarity(rows):
    """Analyze P0/B100k error complementarity per region."""
    # Group by region
    by_region = defaultdict(dict)
    for r in rows:
        by_region[r["region_id"]][r["model"]] = r

    results = []
    for region_id, models in sorted(by_region.items()):
        if "G-P0" not in models or "G-B100" not in models:
            continue

        p0 = models["G-P0"]
        b100 = models["G-B100"]
        source = p0["source_id"]

        # Extract confusion counts
        p0_tp, p0_fp, p0_fn, p0_tn = int(p0["tp"]), int(p0["fp"]), int(p0["fn"]), int(p0["tn"])
        b_tp, b_fp, b_fn, b_tn = int(b100["tp"]), int(b100["fp"]), int(b100["fn"]), int(b100["tn"])
        total = p0_tp + p0_fp + p0_fn + p0_tn

        p0_dice = float(p0["dice"])
        b_dice = float(b100["dice"])
        better_dice = max(p0_dice, b_dice)
        better_model = "P0" if p0_dice >= b_dice else "B100k"

        # Count error patterns (these are exact from full-volume evaluation)
        # P0 errors: FP pixels (p0_fp) and FN pixels (p0_fn)
        # B100k errors: FP pixels (b_fp) and FN pixels (b_fn)
        p0_error_pixels = p0_fp + p0_fn
        b_error_pixels = b_fp + b_fn

        # GT foreground = TP + FN (should be same for both)
        gt_fg = p0_tp + p0_fn
        gt_fg_b = b_tp + b_fn
        assert gt_fg == gt_fg_b, f"GT mismatch: {gt_fg} vs {gt_fg_b} for {region_id}"

        # Without pixel-level overlap we cannot compute exact oracle Dice,
        # but we CAN compute informative bounds:
        #
        # Best case oracle: each pixel uses whichever model is correct
        # - Oracle TP >= max(p0_tp, b_tp)  (if both TPs overlap, just max)
        # - Oracle FP <= min(p0_fp, b_fp)  (only common FPs remain)
        # - Oracle FN <= min(p0_fn, b_fn)  (only common FNs remain)
        #
        # Worst case oracle: errors completely overlap
        # - Oracle TP = max(p0_tp, b_tp)
        # - Oracle FP = max(p0_fp, b_fp) if errors overlap else...
        #
        # For a TIGHT upper bound, assume maximal complementarity:
        oracle_tp_upper = min(gt_fg, p0_tp + b_tp)  # can't exceed GT fg
        oracle_fp_lower = max(0, p0_fp + b_fp - total + gt_fg + max(p0_tp, b_tp))
        oracle_fn_lower = max(0, gt_fg - oracle_tp_upper)

        # Simpler optimistic bound:
        oracle_tp_opt = min(gt_fg, max(p0_tp, b_tp) + abs(p0_tp - b_tp))
        # Actually the tightest optimistic oracle:
        # If errors are completely disjoint, oracle gets all correct pixels from both
        oracle_tp_best = min(gt_fg, p0_tp + (b_tp - max(0, p0_tp + b_tp - gt_fg)))
        # = min(gt_fg, p0_tp + b_tp - max(0, p0_tp + b_tp - gt_fg))
        # = min(gt_fg, min(p0_tp + b_tp, gt_fg)) = min(gt_fg, gt_fg) when sum > gt_fg
        # Hmm, let's be more careful. TP is correctly predicted foreground.

        # P0 correct foreground: p0_tp pixels.  B100k correct foreground: b_tp pixels.
        # Both from same gt_fg total foreground pixels.
        # Union of correct foreground = p0_tp + b_tp - overlap_correct_fg
        # Best case (max complementarity): overlap = max(0, p0_tp + b_tp - gt_fg)
        # So union_correct_fg = min(gt_fg, p0_tp + b_tp - max(0, p0_tp + b_tp - gt_fg))
        #                     = min(gt_fg, min(p0_tp+b_tp, gt_fg)) = min(gt_fg, gt_fg) if sum>=gt_fg
        # That's just gt_fg if both have high recall. That makes sense.
        # But if sum < gt_fg: union = p0_tp + b_tp (no overlap needed)

        oracle_tp_bestcase = min(gt_fg, p0_tp + b_tp)

        # For FP: P0 predicts p0_fp false positives, B100k predicts b_fp.
        # Best case: these are completely different background pixels,
        # but oracle only selects predictions from one model per pixel.
        # In best case, oracle FP = 0 (every FP pixel, the other model was correct = TN)
        # This is overly optimistic but gives the upper bound.

        # For FN: similarly, best case oracle FN = max(0, p0_fn + b_fn - gt_fg)
        oracle_fn_bestcase = max(0, p0_fn + b_fn - gt_fg)
        # = max(0, (gt_fg - p0_tp) + (gt_fg - b_tp) - gt_fg)
        # = max(0, gt_fg - p0_tp - b_tp)

        # Best-case oracle Dice (assuming perfect per-pixel routing):
        den_best = 2 * oracle_tp_bestcase + 0 + oracle_fn_bestcase
        oracle_dice_upper = 2 * oracle_tp_bestcase / den_best if den_best > 0 else 1.0

        # Conservative oracle: assume all FPs overlap (worst case for FP)
        oracle_fp_conservative = min(p0_fp, b_fp)
        oracle_fn_conservative = max(0, gt_fg - oracle_tp_bestcase)
        den_cons = 2 * oracle_tp_bestcase + oracle_fp_conservative + oracle_fn_conservative
        oracle_dice_conservative = 2 * oracle_tp_bestcase / den_cons if den_cons > 0 else 1.0

        # Complementarity index: what fraction of errors are non-overlapping?
        # Using a simple proxy: |errors_P0 - errors_B100k| / max(errors_P0, errors_B100k)
        # More meaningful: potential gain = oracle_upper - better_single
        potential_gain_upper = oracle_dice_upper - better_dice
        potential_gain_conservative = oracle_dice_conservative - better_dice

        results.append({
            "region_id": region_id,
            "source_id": source,
            "p0_dice": p0_dice,
            "b100_dice": b_dice,
            "better_model": better_model,
            "better_dice": better_dice,
            "delta_b100_minus_p0": b_dice - p0_dice,
            "p0_tp": p0_tp,
            "p0_fp": p0_fp,
            "p0_fn": p0_fn,
            "b100_tp": b_tp,
            "b100_fp": b_fp,
            "b100_fn": b_fn,
            "gt_foreground": gt_fg,
            "total_pixels": total,
            "oracle_tp_bestcase": oracle_tp_bestcase,
            "oracle_fn_bestcase": oracle_fn_bestcase,
            "oracle_dice_upper": oracle_dice_upper,
            "oracle_dice_conservative": oracle_dice_conservative,
            "potential_gain_upper": potential_gain_upper,
            "potential_gain_conservative": potential_gain_conservative,
        })

    return results


def summarize_by_source(results):
    """Compute per-source summary statistics."""
    by_source = defaultdict(list)
    for r in results:
        by_source[r["source_id"]].append(r)

    summaries = []
    for source, regions in sorted(by_source.items()):
        n = len(regions)
        p0_dice = sum(r["p0_dice"] for r in regions) / n
        b100_dice = sum(r["b100_dice"] for r in regions) / n
        oracle_upper = sum(r["oracle_dice_upper"] for r in regions) / n
        oracle_cons = sum(r["oracle_dice_conservative"] for r in regions) / n
        gain_upper = sum(r["potential_gain_upper"] for r in regions) / n
        gain_cons = sum(r["potential_gain_conservative"] for r in regions) / n

        # Direction: how many regions is P0 better vs B100k better?
        p0_wins = sum(1 for r in regions if r["better_model"] == "P0")
        b100_wins = n - p0_wins

        summaries.append({
            "source_id": source,
            "n_regions": n,
            "p0_dice": p0_dice,
            "b100_dice": b100_dice,
            "better_mean": max(p0_dice, b100_dice),
            "oracle_dice_upper": oracle_upper,
            "oracle_dice_conservative": oracle_cons,
            "gain_upper_vs_better": gain_upper,
            "gain_conservative_vs_better": gain_cons,
            "p0_wins": p0_wins,
            "b100_wins": b100_wins,
        })

    return summaries


def overall_summary(source_summaries):
    """Compute source-equal-weight overall statistics."""
    n = len(source_summaries)
    return {
        "n_sources": n,
        "total_regions": sum(s["n_regions"] for s in source_summaries),
        "p0_source_eq_dice": sum(s["p0_dice"] for s in source_summaries) / n,
        "b100_source_eq_dice": sum(s["b100_dice"] for s in source_summaries) / n,
        "oracle_upper_source_eq": sum(s["oracle_dice_upper"] for s in source_summaries) / n,
        "oracle_conservative_source_eq": sum(s["oracle_dice_conservative"] for s in source_summaries) / n,
        "gain_upper_vs_better": sum(s["gain_upper_vs_better"] for s in source_summaries) / n,
        "gain_conservative_vs_better": sum(s["gain_conservative_vs_better"] for s in source_summaries) / n,
    }


def main():
    print("=" * 70)
    print("Phase 0: P0/B100k Error Complementarity Analysis")
    print("=" * 70)

    rows = load_mv3_metrics()
    print(f"Loaded {len(rows)} rows from MV-3 metrics (seed 20261014)")

    results = compute_complementarity(rows)
    print(f"Computed complementarity for {len(results)} matched regions")

    # Per-source summary
    source_sums = summarize_by_source(results)
    overall = overall_summary(source_sums)

    # Print source-level results
    print(f"\n{'Source':<20} {'#Reg':>4} {'P0':>7} {'B100':>7} {'OracleU':>8} {'OracleC':>8} {'GainU':>7} {'GainC':>7} {'P0w':>3} {'Bw':>3}")
    print("-" * 90)
    for s in source_sums:
        print(f"{s['source_id']:<20} {s['n_regions']:>4} {s['p0_dice']:>7.4f} {s['b100_dice']:>7.4f} "
              f"{s['oracle_dice_upper']:>8.4f} {s['oracle_dice_conservative']:>8.4f} "
              f"{s['gain_upper_vs_better']:>7.4f} {s['gain_conservative_vs_better']:>7.4f} "
              f"{s['p0_wins']:>3} {s['b100_wins']:>3}")

    print("-" * 90)
    o = overall
    print(f"{'SOURCE-EQ MEAN':<20} {o['total_regions']:>4} {o['p0_source_eq_dice']:>7.4f} {o['b100_source_eq_dice']:>7.4f} "
          f"{o['oracle_upper_source_eq']:>8.4f} {o['oracle_conservative_source_eq']:>8.4f} "
          f"{o['gain_upper_vs_better']:>7.4f} {o['gain_conservative_vs_better']:>7.4f}")

    # Decision gate
    threshold = 0.02
    print(f"\n{'=' * 70}")
    print(f"SARR Feasibility Gate (threshold: +{threshold:.3f} Dice)")
    print(f"  Conservative oracle gain over better single model: {o['gain_conservative_vs_better']:+.4f}")
    print(f"  Upper-bound oracle gain over better single model:  {o['gain_upper_vs_better']:+.4f}")

    if o["gain_conservative_vs_better"] >= threshold:
        decision = "PASS"
        print(f"  Decision: {decision} -- Even conservative estimate exceeds threshold.")
        print(f"  -> Proceed to Phase 1: Router training.")
    elif o["gain_upper_vs_better"] >= threshold:
        decision = "CONDITIONAL_PASS"
        print(f"  Decision: {decision} -- Upper bound exceeds threshold but conservative does not.")
        print(f"  -> Need pixel-level oracle analysis on server to confirm.")
        print(f"  -> Run phase0_oracle_pixelwise.py on server for exact oracle Dice.")
    else:
        decision = "FAIL"
        print(f"  Decision: {decision} -- Insufficient complementarity.")
        print(f"  -> Pivot to Scale-Aware Adapter (backup plan).")

    # Highlight key sources
    print(f"\nKey sources for SARR value:")
    for s in sorted(source_sums, key=lambda x: x["gain_conservative_vs_better"], reverse=True):
        marker = "+" if s["b100_dice"] > s["p0_dice"] else "-"
        print(f"  {s['source_id']:<20} B100-P0={s['b100_dice']-s['p0_dice']:+.4f} {marker}  "
              f"Oracle gain: {s['gain_conservative_vs_better']:+.4f}")

    # Save results
    with open(OUT_DIR / "phase0_complementarity.json", "w") as f:
        json.dump({
            "description": "Phase 0 SARR feasibility: P0/B100k error complementarity from MV-3 confusion counts",
            "seed": 20261014,
            "decision": decision,
            "threshold": threshold,
            "overall": overall,
            "per_source": source_sums,
            "per_region": results,
        }, f, indent=2)

    # Save CSV for easy inspection
    with open(OUT_DIR / "phase0_per_region.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(OUT_DIR / "phase0_per_source.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=source_sums[0].keys())
        writer.writeheader()
        writer.writerows(source_sums)

    print(f"\nResults saved to {OUT_DIR}")
    return decision


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if decision != "FAIL" else 1)
"""
Phase 0 companion script for pixel-level oracle analysis.
This script is designed to run on the SERVER where prediction NPY files exist.

It computes the exact per-pixel oracle Dice by loading both P0 and B100k
full-volume predictions and computing per-pixel optimal selection.

Deploy to server:
    scp phase0_oracle_pixelwise.py root@192.168.203.97:/opt/data/3_117/tanghr/codex/

Usage on server:
    python phase0_oracle_pixelwise.py \\
        --mv3-root /opt/data/3_117/tanghr/codex/mv_generalist_v2_han_fullgt_20261014 \\
        --out /opt/data/3_117/tanghr/codex/sarr_phase0_oracle/
"""
