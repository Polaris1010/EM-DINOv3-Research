"""Unblind only the 20 user-reviewed pairs; preserve the remaining key."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "branch_S/blind_review_raw_20261011"
RESPONSES = OUT / "human_review_P001_P020_overlay_20261011.csv"


def main() -> None:
    with RESPONSES.open(newline="", encoding="utf-8-sig") as stream:
        responses = list(csv.DictReader(stream))
    ids = [r["pair_id"] for r in responses]
    assert len(ids) == len(set(ids)) == 20 and set(ids) == {f"P{i:03d}" for i in range(1, 21)}
    with (RUN / "branch_S/blind_review/answer_key_CONCEALED.csv").open(newline="") as stream:
        key = {r["pair_id"]: r for r in csv.DictReader(stream) if r["pair_id"] in ids}
    assert len(key) == 20
    results = []
    sources = defaultdict(lambda: {"total": 0, "judgeable": 0, "agree": 0, "unclear": 0})
    count = Counter()
    for row in responses:
        pair = row["pair_id"];choice = row["human_choice_A_B_unclear"].strip()
        assert choice in {"A", "B", "unclear"}
        item = key[pair];winner = item["score_winner"]
        assert winner in {"A", "B"}
        count[choice] += 1
        judged = choice != "unclear";agree = choice == winner if judged else None
        source = sources[item["source"]];source["total"] += 1
        source["judgeable"] += int(judged);source["agree"] += int(agree is True);source["unclear"] += int(not judged)
        results.append({"pair_id": pair, "source": item["source"], "human_choice": choice,
                        "algorithm_choice_cut_mean": winner, "agree": agree})
    judged = count["A"] + count["B"];correct = sum(r["agree"] is True for r in results)
    p = correct / judged;z = 1.959963984540054
    center = (p + z*z/(2*judged))/(1+z*z/judged)
    half = z*math.sqrt(p*(1-p)/judged+z*z/(4*judged*judged))/(1+z*z/judged)
    result = {"status": "complete_pilot_audit", "panel_version": "original 20261002 SAM-overlay P001-P020",
              "user_statement": "Blue/cyan blocks were confusing; user judged EM and crop boxes while deliberately ignoring them",
              "total": 20, "choices": dict(count), "judgeable": judged, "judgeable_fraction": judged / 20,
              "algorithm_agreement_count": correct, "algorithm_disagreement_count": judged-correct,
              "agreement_on_judgeable": p, "wilson95_pair_level": [center-half, center+half],
              "sources": dict(sorted(sources.items())), "pairs": results,
              "gate_decision": "NO_FINAL_PASS_FAIL_FROM_20_PAIR_OVERLAY_PILOT",
              "interpretation": "Judgeable-only pilot agreement; overlay visibility and small sample remain limitations. Unclear rows are excluded from agreement denominator, not counted as errors. A/B frequencies alone do not establish algorithm accuracy."}
    (OUT / "human_overlay20_summary.json").write_text(json.dumps(result, indent=2))
    with (OUT / "human_overlay20_unblinded.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    state_path = RUN / "STATUS.json";state = json.loads(state_path.read_text())
    state["S_human_pilot"] = "20_ORIGINAL_OVERLAY_PAIRS_REVIEWED_PENDING_RAW_CONFIRMATION"
    state["S"] = "QUANTITATIVE_COMPLETE_HUMAN_PILOT_AVAILABLE_FINAL_GATE_PENDING"
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
