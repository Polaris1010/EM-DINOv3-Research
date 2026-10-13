# MV-2 推理协议纠正

本轮严格沿用 MV-1 已实现的推理协议，TTA=OFF。

- 直接调用原始 `mv_generalist_stage1_20261013/src/mv_probe.py::sliding`。
- 256×256 滑窗，128 步长，tile batch 16；预处理、精度、概率融合及 0.5 阈值沿用 MV-1。
- 复用 P0、B100k、B150k × 三个 decoder seeds 的已有 9 个 20k-step Primus checkpoint。仅推理，不训练 encoder 或 decoder。
- 首批 `guay21_vol1`、`jiang25_vol2`、`muller21_high_c3`、`muller21_low_c2` 共 36 条既有指标直接复制。代码不对这四个体积调用滑窗推理，也不重新计算距离指标。
- 此前误启动的四翻转任务 `mv-dev-v2-validation-20261013.service` 已停止，保留在独立旧目录 `mv_dev_v2_expansion_20261013`。该任务的预测和指标不进入本轮结果。
- 可以复用其与 TTA 无关、区域和数据元信息哈希一致的原图/真值缓存。

本轮任务为 `mv-dev-v2-notta-20261013.service`，使用 GPU 1–3。完成后由 CPU 汇总进程生成所需逐体积指标、来源对照、配对差值、失效分析和研究决策。

实际清单纳入 188 个开发体积/区域、12 个来源；官方验证与探索性开发分别标明，正式 test 不纳入。计划产生 1,692 条模型/seed/区域指标，其中 36 条为 MV-1 直接复用，1,656 条来自新增区域。

本文件确认执行协议，不代表全量验证已完成。真实运行状态见服务器 `STATUS.json`；本地同名文件为下载时的快照。全量完成与否以 `DELIVERY_READY.json` 为准。

## 本次续跑修复

任务在 HD95 的 `torch.quantile` 处理超过 16,777,216 个表面距离时中断。只对超出该上限的 CPU 距离向量使用 NumPy 的线性 95% 分位数；原始 MONAI 的表面提取、物理距离、双向最大值聚合和 NSD 继续使用。较小向量仍调用原函数。兼容检查复现了原上限，并验证了解析分位数和临时绑定恢复。

错误现场及原脚本保存在服务器 `evidence/quantile_failure_original`，兼容检查结果为 `evidence/hd95_size_limit_fix.json`。已完成的推理与指标 JSON 均跳过，只有缺少完成记录的中断区域继续推理。四个 MV-1 体积仍直接复用。
