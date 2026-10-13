# DINOv3 电镜表征诊断：实时交付

报告：

- [decoder_architecture_audit.md](decoder_architecture_audit.md)
- [dinov3_stage_completion_audit.md](dinov3_stage_completion_audit.md)
- [representation_diagnosis_preflight.md](representation_diagnosis_preflight.md)

实时服务器工作目录：`/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013`。

本地包含服务器 source / evidence / smoke 原件、诊断代码和三份报告。历史预审目录与所有原实验文件保留。原件哈希、当前 GPU / 网站状态和权重测试结果均可检查。

## 代码入口

`scripts/diagnosis.py`：FrozenReadout / LinearProbe / PrimusHead / fit_readout / dense_diagnostic / resize_view / exposure_rows / run_features。正式 runner 要求锁定的训练或开发 ROI，包含 image / label / valid mask 文件哈希、acquisition、annotation evidence、nm spacing / bounds；拒绝 test/external/unknown split。

`scripts/server_factory.py`：strict 真实 DINOv3 encoder 和服务器 Primus 构造器，校验 source / resolved config；调用方负责校验 checkpoint hash。加载 teacher 时校验 step 和正式 SSL manifest。

`scripts/live_smoke.py`：本轮限定技术测试。只允许 CUDA_VISIBLE_DEVICES=3，GPU3 必须无外部进程，网站必须返回 200；三模型各两种 head 两步优化，六种分辨率视图，校验冻结和输入一致性。不含 SSL 入口，不应将此 smoke 当作完整性能评价。

`evidence/diagnosis_plan.unlocked.json`：真实权重 hash 已补齐，正式 ROI 未锁定，因此仍被 validator 拒绝。此状态是尚缺 acquisition / annotation 证据，不是 SSH 认证问题。

正式锁定后，feature 诊断入口如下；当前 unlocked plan 不应执行：

```bash
CUDA_VISIBLE_DEVICES=3 /opt/data/3_117/tanghr/envs/meddinov3/bin/python scripts/diagnosis.py \
  --plan <locked_development_plan.json> --factory scripts/server_factory.py \
  --output <new_output_directory> --device cuda:0 --execute-features
```

cuda:0 在这里是已屏蔽后的 physical GPU3，不是服务器 GPU0。完整多 seeds Primus 应另用历史 label1 probe，以保留其 sampler / sliding / TTA。

## 本地交付一致性检查

```powershell
F:\Anaconda\python.exe artifacts/representation_diagnosis_live_20261013/verify_delivery.py
```

该检查读取 JSON / source hashes / 报告链接并汇总 smoke，不启动模型或训练。完整脚本与真实测量输出已保存，见 verification_results.json。GPU3 的约 2.7 GiB peak 只适用于此次冻结 smoke，不能用于估计 HRA backward batch。
