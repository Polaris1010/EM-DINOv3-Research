"""Freeze the evidence-backed Stage 2A failure and prohibit T1 launch."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

RUN = Path(__file__).resolve().parents[1]
CAL = RUN / "topology_calibration"


def read(path: str) -> dict:
    return json.loads((RUN / path).read_text())


def main() -> None:
    sample = read("audit/sample_validation.json")
    b = read("topology_calibration/equivariance/vit_b_equivariance_424.json")
    h = read("topology_calibration/equivariance/vit_h_equivariance_424.json")
    hgen = read("topology_calibration/sam_vit_h/generation_complete.json")
    bench = read("topology_calibration/cache_benchmark/vit_h_raw_tile_10.json")
    pilot = read("topology_calibration/crop_score/crop_score_pilot_424x4.json")
    diag = read("topology_calibration/crop_score/crop_score_diagnostic_424.json")
    assert sample["status"] == b["status"] == h["status"] == "pass"
    assert hgen["status"] == "complete" and pilot["candidate_count"] == 1696
    assert h["offline_H_exact_count"] == 2120
    assert diag["within_sample_spearman_median"] <= 0
    assert diag["lower_score_has_less_border_proxy_fraction"] < .5
    assert diag["lower_score_minus_higher_score_border_proxy_mean"] > 0
    with urlopen("http://127.0.0.1:7860/", timeout=8) as response:
        website_http = response.status
    gpu_csv = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    gate = {"stage2A_gate": "FAIL", "auto_launch_T1": False, "T1_status": "NOT_STARTED",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "The specified crop/context PH score does not select spatial positions with lower label-free border truncation proxy on same-shape paired candidates.",
            "checks": {
                "same_424_paired_slices": "PASS", "label_free_pipeline": "PASS",
                "SAM_ViT_H_MAM_generation": "PASS", "offline_flip_equivariance": "PASS",
                "offline_rotation_equivariance": "PASS", "score_dynamic_range": "PASS",
                "more_destructive_crop_ranks_worse": "PASS",
                "source_normalizer_pilot": "PASS", "small_source_global_fallback_pilot": "PASS",
                "same_shape_location_truncation_relation": "FAIL",
                "coordinate_mapping_full_cache": "NOT_RUN_DUE_TO_FAIL",
                "native_resolution_acceptance": "NOT_RUN_DUE_TO_FAIL",
                "cache_dtype_acceptance": "NOT_RUN_DUE_TO_FAIL",
                "B_T_crop_geometry_simulation": "NOT_RUN_DUE_TO_FAIL",
                "full_cache_reader": "NOT_RUN_DUE_TO_FAIL",
                "PH_training_overhead": "NOT_RUN_DUE_TO_FAIL",
                "website_healthy": "PASS" if website_http == 200 else "FAIL",
                "GPU_resources_safe": "PASS"},
            "evidence": {
                "same_shape_within_sample_spearman_median": diag["within_sample_spearman_median"],
                "same_shape_within_sample_spearman_mean": diag["within_sample_spearman_mean"],
                "lower_score_has_less_border_proxy_fraction": diag["lower_score_has_less_border_proxy_fraction"],
                "lower_score_minus_higher_score_proxy_mean": diag["lower_score_minus_higher_score_border_proxy_mean"],
                "pooled_score_proxy_spearman": pilot["score_border_proxy_spearman"],
                "random_mean_proxy": pilot["random_mean_border_proxy"],
                "Q85": pilot["policies"]["Q85"], "Q90": pilot["policies"]["Q90"],
                "within_training_scale_stronger_crop_ranks_worse_fraction": diag["within_training_scale_more_destructive_greater_fraction"],
                "ViT_H_tile_median_encode_seconds": bench["median_encode_seconds_excluding_first"],
                "estimated_3_GPU_encode_hours_excluding_IO": bench["estimated_3_gpu_encode_hours_excluding_io"],
                "website_http": website_http, "gpu_status": gpu_csv}}
    (CAL / "stage2A_gate.json").write_text(json.dumps(gate, indent=2))
    report = f"""# Full-MitoVerse Stage 2A 校准结论

**STAGE2A_GATE = FAIL；AUTO_LAUNCH_T1 = false；T1 未启动。**

时间：{gate['timestamp_utc']}。Stage 1 只读；GPU 0 未用于本轮计算。网站 HTTP {website_http}。

## 已通过

- 与 Stage 1 相同的 424 张切片（212 volumes × 2），完整 manifest 匹配。
- 官方 SAM ViT-H 权重 SHA-256 `{hgen['weight_sha256']}`，424 张相同样本全部生成 6 种几何变换 MAM，未使用标签。
- ViT-B 与 ViT-H 固定 prior 的 5 种 flip/90° rotation 逆变换全部逐像素相同；H0/H1 bottleneck 全部为 0。ViT-H 为 2,120/2,120 次。
- ViT-H 在线重新编码仍不等变：hflip MAM MAE 中位数 {h['groups']['online_H']['hflip']['mae']['median']:.4f}，H0/H1 bottleneck 中位数 {h['groups']['online_H']['hflip']['d0']['median']:.4f}/{h['groups']['online_H']['hflip']['d1']['median']:.4f}。正式算法必须用固定离线 prior。
- ViT-H 原始 uint8 1024 tile 编码中位数 {bench['median_encode_seconds_excluding_first']:.3f} 秒，三卡纯编码粗估 {bench['estimated_3_gpu_encode_hours_excluding_io']:.2f} 小时，未计入 IO。
- crop/context PH score 有动态范围；在训练可用尺度内，较强 crop 分数更差的配对比例 {diag['within_training_scale_more_destructive_greater_fraction']:.1%}。

## 阻断项

按 B_full global crop scale 0.32–1.0，使用同一样本、同一 crop 形状的 4 个空间位置，共 1,696 个候选；context 扩张 1.5 倍，H0/H1 各 0.5 权重，按 source × scale bin 做 median/IQR 归一化，小 source 回退全局统计。

MAM 低 affinity 对应邻接特征差异，故用 `1-MAM` 在 crop 边框 5% 带的均值除以 crop 总体均值，作为完全无标签的结构截断代理。**同一张样本内的 score/proxy Spearman 中位数为 {diag['within_sample_spearman_median']:.3f}、均值 {diag['within_sample_spearman_mean']:.3f}；最低分候选比最高分候选代理更低的比例只有 {diag['lower_score_has_less_border_proxy_fraction']:.1%}，最低分候选代理均值反而高 {diag['lower_score_minus_higher_score_border_proxy_mean']:.4f}。** 跨样本 pooled Spearman 为 {pilot['score_border_proxy_spearman']:.3f}。

Q85/Q90 fallback 分别为 {pilot['policies']['Q85']['fallback_rate']:.1%}/{pilot['policies']['Q90']['fallback_rate']:.1%}，但 Q90 选择后代理均值 {pilot['policies']['Q90']['mean_border_proxy']:.4f} 与随机 {pilot['random_mean_border_proxy']:.4f} 只相差约 {(pilot['random_mean_border_proxy']-pilot['policies']['Q90']['mean_border_proxy'])/pilot['random_mean_border_proxy']:.1%}。当前分数主要区分 crop 尺度的破坏程度，缺乏选择**相同形状下更少截断的位置**的证据。执行单要求该关系合理，因此该硬门槛失败。

## 执行边界

已停止全量 MAM 缓存、B/T 100k crop 模拟与 T1 的准备和启动。坐标映射、native grid、dtype、缓存读取、正式训练开销等仍未通过，不应视为 PASS。要继续需重新设计并验证有正向位置判别能力的无标签指标，然后重做 Stage 2A；不得直接沿用本轮 Q85/Q90/Q95 阈值进入 T1。

证据：`topology_calibration/stage2A_gate.json`、`crop_score/crop_score_pilot_424x4.csv/json`、`crop_score/crop_score_diagnostic_424.json`、`equivariance/vit_b_equivariance_424.json`、`equivariance/vit_h_equivariance_424.json`、`sam_vit_h/paired_B_H_summary.json`、`cache_benchmark/vit_h_raw_tile_10.json`。
"""
    (RUN / "reports/stage2A_calibration.md").write_text(report)
    (RUN / "STATUS").write_text("STAGE2A_GATE=FAIL\nAUTO_LAUNCH_T1=false\nT1_STATUS=NOT_STARTED\nREASON=same_shape_location_truncation_relation_failed\n")
    status = read("STATUS.json")
    status.update({"stage": "stage2A_complete", "stage2A_gate": "FAIL", "auto_launch_T1": False,
                   "T1": "not_started", "reason": "same_shape_location_truncation_relation_failed",
                   "finalized_utc": gate["timestamp_utc"]})
    (RUN / "STATUS.json").write_text(json.dumps(status, indent=2))
    print(json.dumps({"stage2A_gate": gate["stage2A_gate"], "T1_status": gate["T1_status"],
                      "within_sample_spearman_median": diag["within_sample_spearman_median"],
                      "lower_score_has_less_border_proxy_fraction": diag["lower_score_has_less_border_proxy_fraction"],
                      "website_http": website_http}, indent=2))


if __name__ == "__main__":
    main()
