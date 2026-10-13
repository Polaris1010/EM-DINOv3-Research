# DINOv3 representation audit delivery

入口：[preflight](representation_diagnosis_preflight.md)。本目录为独立交付，历史资产保持原状。

- [decoder 审计](decoder_architecture_audit.md)
- [Gram/HRA 审计](dinov3_stage_completion_audit.md)
- [参数摘要](model_parameter_summary.json)
- [历史 P0/B 对照证据](historical_probe_evidence.csv)
- [开发数据暴露记录](diagnosis_data_exposure_records.json)
- [checkpoint 可用性](checkpoint_availability.json)
- [诊断 plan 模板](diagnosis_plan.template.json)：故意不可执行，缺物理 ROI 和 100k teacher digest。

实际原始与预处理标签的对应关系、valid mask、acquisition/ROI nm 坐标必须先核验。新 ROI 输入使用二维 NPY 图像、标签和有效 mask。binary_mito 的 1 为前景；instance_mito 的所有正整数是线粒体实例，负值忽略。输入需来自 train/dev；test/external/quarantine 被诊断工具拒绝。

工具角色：`tools/diagnosis.py` 提供 readout/frozen linear、dense、分辨率与配对诊断；`tools/server_factory.py` 导入实际服务器 DINOv3 与 Primus；`tools/source_smoke.py` 为 CPU 源码验证；`tools/collect_live_evidence.py` 只读收集现场证据，不 import torch，不访问 GPU。

在服务器端只读采集，stdout 重定向到**新的**输出文件，roots 必须覆盖实际历史 run，不输入网站目录。可单独添加 Primus、实际 LayerNormNd 源文件。默认只列 checkpoint，hash 需显式 --hash-checkpoints（增加磁盘读取，不改文件）：

```bash
python collect_live_evidence.py \
  --root /opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120 \
  --root /opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002 \
  --root /opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012 \
  --source-file /opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py
```

plan 的 status 仅在真实输入及元数据验证后设为 locked_development_diagnosis。执行 features 前还需补齐 checkpoint SHA、factory hash、resolved config/hash、正确 intensity range、rgb normalization 和 ROI entries。默认 CPU；GPU 模式仅允许 CUDA_VISIBLE_DEVICES 单个物理 1、2 或 3，再指定 cuda:0。可见 index0 此时映射到允许的物理卡；未明确屏蔽物理 GPU0 时工具拒绝 GPU 执行。

```bash
python diagnosis.py --plan audited_dev_plan.json
python diagnosis.py --plan audited_dev_plan.json --factory server_factory.py \
  --execute-features --device cpu --output NEW_OUTPUT_DIRECTORY
```

`fit_readout` 是可复用的 detached-feature head fitter，没有自动训练 CLI。历史完整 Primus 三 seed 重跑应复用原 label1 frozen probe 的 loader/采样/滑窗/TTA；paired_roi_cached_features_v1 的结果必须另存和单独命名。

本轮没有完成新真实 DINOv3 权重实验；合成 smoke 仅验证代码。详细未证实项见 preflight。
