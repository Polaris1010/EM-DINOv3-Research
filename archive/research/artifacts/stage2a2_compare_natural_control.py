"""Summarize the matched official-natural/B_full frozen evaluations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "branch_E/control_20261011"


def main() -> None:
    match = json.loads((OUT / "protocol_match.json").read_text())
    assert match["status"] == "pass"
    datasets = ("Dataset111_CMCC", "Dataset131_CMCT")
    rows, cases, audit = [], [], []
    for ds in datasets:
        natural_dir = OUT / f"runs/official_natural_{ds}"
        assert json.loads((natural_dir / "result.json").read_text())["status"] == "complete"
        natural = json.loads((natural_dir / "metrics.json").read_text())
        base = json.loads((RUN / f"branch_E/runs/B_full_{ds}/metrics.json").read_text())
        assert natural["status"] == base["status"] == "complete"
        for field in ("split_sha256", "seed", "steps"):
            assert natural[field] == base[field], field
        assert natural["steps"] == 3000 and natural["seed"] == 20260907
        assert natural["decoder_initial_sha256"] == match["decoder_initial_sha256"]
        assert natural["frozen_encoder"] and natural["encoder_sha256"] == match["official_sha256"]
        n, b = natural["datasets"][ds], base["datasets"][ds]
        assert n["slice_stride"] == b["slice_stride"] == 1 and n["tta"] and b["tta"]
        assert set(n["cases"]) == set(b["cases"])
        rows.append({"dataset": ds, "val_cases": len(n["cases"]), "official_macro_dice": n["macro_dice"],
                     "B_full_macro_dice": b["macro_dice"], "B_minus_official_macro_pp": 100 * (b["macro_dice"] - n["macro_dice"]),
                     "official_micro_dice": n["micro"]["dice"], "B_full_micro_dice": b["micro"]["dice"],
                     "official_selected_decoder_step": n["selected_step"], "B_full_selected_decoder_step": b["selected_step"]})
        for case in n["cases"]:
            nc, bc = n["cases"][case], b["cases"][case]
            assert nc["tp"] + nc["fn"] == bc["tp"] + bc["fn"]
            assert nc["tn"] + nc["fp"] == bc["tn"] + bc["fp"]
            cases.append({"dataset": ds, "case": case, "official_dice": nc["dice"], "B_full_dice": bc["dice"],
                          "B_minus_official_pp": 100 * (bc["dice"] - nc["dice"])})
        audit.append({"dataset": ds, "split_sha256": natural["split_sha256"], "control_source_sha256": natural["source_sha256"],
                      "original_B_source_sha256": base["source_sha256"], "decoder_initial_sha256": natural["decoder_initial_sha256"],
                      "val_label_voxel_counts_identical": True})
    for name, data in (("comparison.csv", rows), ("case_comparison.csv", cases)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader();writer.writerows(data)
    summary = {"status": "complete", "protocol_match": "pass", "datasets": rows, "cases": cases, "audit": audit,
               "interpretation": "single-seed frozen decoder validation comparison, not independent test performance; SSL/downstream raw-data overlap not yet audited",
               "P": "paused", "long_training": "prohibited"}
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2))
    report = ["# 官方自然图像权重与 B_full 的冻结评测对照", "", "日期：2026-10-11。encoder 全部冻结，只训练相同的 Primus decoder 与 gates，各 3,000 step，seed=20260907。", "",
              "| 数据集 | 验证例数 | 官方自然权重宏 Dice | B_full 宏 Dice | B_full−官方（百分点） |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['dataset']} | {row['val_cases']} | {row['official_macro_dice']:.4f} | {row['B_full_macro_dice']:.4f} | {row['B_minus_official_macro_pp']:+.2f} |")
    report.extend(["", "配对控制已验证：decoder 初始参数相同，采样、loss、训练循环、seed、数据 split 相同；B_full 兼容性 batch 输出差为 0。两个 encoder 均严格加载 368 个 backbone keys。完整评测使用同一验证体积、原尺寸恢复、四翻转；验证标签正负 voxel 总数一致。", "",
                   "两个模型各自根据 stride-20 验证 Dice 选 decoder checkpoint，再做完整验证。这是同一协议下的验证集比较；两项任务分别只有 3 例和 2 例，且仅一个 seed，不能据此声称统计显著或未见 source 的泛化改善。SSL 与下游原图体积重叠仍需单独审计。", "",
                   "逐例结果：case_comparison.csv；协议检查与依赖文件 hash：protocol_match.json；原始 control metrics 位于 runs/official_natural_<dataset>/metrics.json。P 保持暂缓，未启动任何 SSL 长训。"])
    (OUT / "COMPARISON_zh.md").write_text("\n".join(report) + "\n")
    state_path = RUN / "STATUS.json"
    state = json.loads(state_path.read_text())
    state["E"] = "COMPLETE_MATCHED_B_FULL_AND_OFFICIAL_NATURAL_CMCC_CMCT"
    state["S_raw_review"] = "100_RAW_PANELS_READY_20_PAIR_TRIAL_PENDING_HUMAN_REVIEW"
    state["P"] = "PAUSED_BY_USER_PENDING_MEASUREMENT_DEFINITION"
    state["long_training"] = "PROHIBITED"
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
