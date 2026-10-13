import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent / "mitoverse_full_stage2a2_result"
with (ROOT / "same_shape_scores_424.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
by_sample = defaultdict(list)
by_source = defaultdict(list)
for row in rows:
    by_sample[int(row["sample_index"])].append(row)
    by_source[row["source"]].append(row)

metrics = ["old_pd_score", "cut_mean", "persist_exclude_largest"]
summary = {"samples": len(by_sample), "sources": {}, "rank_agreement": {}, "eligible_pairs": {}}
for source, source_rows in sorted(by_source.items()):
    sample_ids = {int(r["sample_index"]) for r in source_rows}
    eligible = sum(sum(r["all_four_sides_observed"] == "True" for r in by_sample[i]) >= 2 for i in sample_ids)
    ranges = [max(float(r["cut_mean"]) for r in by_sample[i]) - min(float(r["cut_mean"]) for r in by_sample[i]) for i in sample_ids]
    summary["sources"][source] = {"samples": len(sample_ids), "eligible_pair_samples": eligible, "median_cut_mean_range": float(np.median(ranges)), "nonzero_range_fraction": float(np.mean(np.array(ranges) > 1e-6))}
for other in metrics:
    if other == "cut_mean":
        continue
    rank = []
    for group in by_sample.values():
        a = [float(r["cut_mean"]) for r in group]
        b = [float(r[other]) for r in group]
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or len(set(a)) < 2 or len(set(b)) < 2:
            continue
        rank.append(float(spearmanr(a, b).statistic))
    summary["rank_agreement"][other] = {"n": len(rank), "median_spearman": float(np.median(rank)), "fraction_positive": float(np.mean(np.array(rank) > 0))}

eligible_by_source = defaultdict(list)
for i, group in by_sample.items():
    complete = [r for r in group if r["all_four_sides_observed"] == "True"]
    if len(complete) >= 2:
        scores = sorted(float(r["cut_mean"]) for r in complete)
        eligible_by_source[group[0]["source"]].append((i, scores[-1] - scores[0]))
summary["eligible_pairs"] = {source: len(pairs) for source, pairs in sorted(eligible_by_source.items())}
print(json.dumps(summary, indent=2))
