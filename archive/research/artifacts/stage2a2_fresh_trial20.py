"""Package 20 still-blinded raw pairs after the original first 20 were reviewed."""
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

from raw_trial_panels import select_trial

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "branch_S/blind_review_raw_20261011"


def main():
    with (OUT / "human_review_P001_P020_overlay_20261011.csv").open(newline="") as stream:
        reviewed = {r["pair_id"] for r in csv.DictReader(stream)}
    with (RUN / "branch_S/blind_review/answer_key_CONCEALED.csv").open(newline="") as stream:
        fields = ("pair_id", "sample_index", "source", "A_candidate", "B_candidate")
        keys = [{f: r[f] for f in fields} for r in csv.DictReader(stream) if r["pair_id"] not in reviewed]
    chosen = select_trial(keys)
    assert not (reviewed & {r["pair_id"] for r in chosen})
    name = "human_review_trial20_fresh.csv"
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair_id", "human_choice_A_B_unclear", "unclear_reason_optional", "note_optional"])
        writer.writeheader();writer.writerows({"pair_id": r["pair_id"]} for r in chosen)
    readme = "FRESH_TRIAL_README_zh.md"
    (OUT / readme).write_text("# 新一批无覆盖原图试审\n\n这一包排除了已审核并解盲的 P001–P020，共 20 组，覆盖 13 个 source。算法分数与胜者仍隐藏。\n\n请按图片左上角的实际 pair_id 填表，不要重新编号为 P001–P020。先读 REVIEW_RULES_zh.md。A/B 配对及黄框位置沿用原实验，只有蓝青色覆盖层被移除。\n\n这批用于确认先导 40% 一致率的负面信号是否在无覆盖原图中仍存在；完成前不宣布 S PASS，不增加长训预算。\n")
    with zipfile.ZipFile(OUT / "blind_raw_trial20_fresh.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for r in chosen:
            for folder in ("panels", "context"):
                archive.write(OUT / folder / (r["pair_id"] + ".png"), folder + "/" + r["pair_id"] + ".png")
        for file in (name, readme, "REVIEW_RULES_zh.md"):
            archive.write(OUT / file, file)
    info = {"status": "ready_not_reviewed", "pairs": 20, "sources": dict(Counter(r["source"] for r in chosen)),
            "pair_ids": [r["pair_id"] for r in chosen], "excluded": sorted(reviewed),
            "selection": "same deterministic source coverage rule; already reviewed IDs excluded; no score or visual selection",
            "zip_bytes": (OUT / "blind_raw_trial20_fresh.zip").stat().st_size}
    (OUT / "fresh_trial20_manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
