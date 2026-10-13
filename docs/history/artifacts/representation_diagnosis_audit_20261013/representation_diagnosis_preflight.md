# Representation Diagnosis Preflight · 2026-10-13

**历史核验完成；本地诊断实现及 CPU smoke 通过；新的真实权重诊断待服务器执行。** 本轮没有可用的服务器登录会话或凭据，也未取得 P0/B encoder 二进制、CMCC/CMCT 原始数据和可追溯物理 ROI。缺失项标为 UNVERIFIED，不用合成结果替代 EM 实验。

## 已确认事实

1. P0/B100k/B150k 的相同历史 frozen probe 已在 CMCC、CMCT 完成，均是 seed 20260907 / 3,000 steps；P0 不是计划状态。
2. 实际 head 是含四个启用 LaSt_FrequencyGate 的 modified Primus，源码与当时服务器依赖 hash 相同；其 SSL backbone 不含该模块。必须披露 gates，单独命名 ungated Primus 新对照。
3. 本项目 B100k/B150k 的 Gram 关闭、views 固定 256/112；没有在这两轮中执行独立 HRA。
4. 已有结果是 mitochondria-only，小样本开发验证；SD 是体积间波动，物理距离单位未校正。标注完整性、acquisition 隔离仍未获证实。
5. B_full 为 all-MitoVerse transductive 资产，干净数据治理仍为 NO_GO；本轮不改变历史 checkpoint、manifest、结果或 Stage2A FAIL。

分别见 [decoder 审计](decoder_architecture_audit.md)、[阶段审计](dinov3_stage_completion_audit.md)、[参数摘要](model_parameter_summary.json) 与 [历史对照表](historical_probe_evidence.csv)。

## 实现范围及验证边界

| 工作 | 已实现/验证 | 真实 EM 状态 |
|---|---|---|
| 相同 gated Primus | 精确源码 class body 构造、真实 1024 head、CPU 前向/梯度、encoder freeze | 历史单 seed 六组已完成；新增三 seeds 未运行 |
| Frozen Linear Probe | 最后一层 1×1 Conv 或明确登记的四层拼接、双类别输出、冻结特征、valid-mask CE+Dice | 合成 matched fitter smoke 通过；真实训练未运行 |
| Dense patch diagnostic | 四层；pure/valid patch、fg/bg centroid、方差、有效秩、同实例/边界邻接 cosine | 合成目标/背景测试通过；真实输出 UNVERIFIED |
| 256/384/512 | fixed_fov 和 native_fov 两种模式、记录 crop bounds、token 数和原生信息变化 | 合成转换及 validity 测试通过；真实模型分辨率收益 UNVERIFIED |
| Paired source/physical ROI | 同一 plan 三模型、同图像 hash、acquisition/spacing/ROI 必填、输入 shape 与 nm bounds 一致性校验 | 没有锁定真实 ROI；模板不可执行 |
| Exposure | 既有验证 case 的方法开发暴露表；新 plan 逐模型/ROI/来源记录 | 未验证上游自然权重逐图重叠或全部标注完整性 |

`tools/diagnosis.py` 不含 SSL 启动入口。`FrozenReadout` 在每次 train() 后强制 encoder.eval()，特征 no_grad/detach；`PrimusHead` 使用传入的真实 Primus up_projection 和 gates，不静默去掉历史 gates。`fit_readout` 仅训练 head，预算限制 1–3,000 steps，并对 P0/B 两组共享 seed/预算/有效标签规则。缓存 ROI fitter 是**新的诊断协议**：不复现旧 nnU-Net 的前景采样、滑窗或 TTA，不能与旧 metrics 混算。完整历史三 seed 重跑仍应复用原 label1 probe 和已记录的采样/验证流程。

`tools/server_factory.py` 提供实际 DINOv3 resolved config / strict checkpoint adapter 和真实 Primus factory；所有路径及 hash 由 plan 明示。服务器模型导入与实际 weights smoke **UNVERIFIED**。当前模板故意为 `UNVERIFIED_ROIS_NOT_LOCKED` 且 rois=[]，100k teacher digest、真实强度范围与物理 ROI 必须先核实；验证器会拒绝该模板。不得为了让它通过而填写猜测值。

Dense cosine 只比较同一步距的 x/y 邻接 patch，在方向和目标像素面积分层内报告均值/数量与 delta；再由 source、acquisition、物理尺度分层分析。binary labels 只允许称 same foreground class，不能称 same mitochondrion。有效秩用 singular-value entropy，未混用平方谱熵定义。pure patch 筛选遗漏弱边界/小目标，centroid L2 不是线性可分性的证明；不单凭某一个诊断量下结论。

fixed_fov：固定原生 ROI，重采样到三种输入大小；可能改变有效采样密度，但插值不会创造信息。native_fov：居中提取真实 256/384/512 像素，保持 native spacing，物理视野随之改变。这两轴不能混算成一个 HRA 增益。当前实现输出 feature 诊断，没有冒充已经完成滑窗/TTA 对齐后的 NSD/HD95。后续分割分辨率评价必须统一 ROI、预测还原网格、valid mask 和实际 spacing。

## 实测检查

CPU decoder smoke：PASS，真实 head 1024 维，合成 32×32 输入；encoder 哈希保持一致且无梯度，gates/up_projection 有限梯度。参数构造总计 46,691,082。源码 harness 使用替代 framework base / 初始化 helper 及捕获的参考 LayerNormNd，不等于服务器完整环境复现。

单元测试覆盖冻结与同 seed 初始化、ignore 梯度、目标边界与秩、mixed/ignored patch 排除、分辨率双轴、禁止 test/external/unknown split、路径 hash/physical shape、一致的三模型 readout-only fitter。检查结果记录于 `verification_results.json`。历史输入哈希保护记录于 `historical_input_integrity.json` / `preservation_check.json`。

## 待补证据与最有价值的下一轮

1. 用只读 collector 取得当前 primus.py、LayerNormNd、resolved YAML、checkpoint inventory 与 step logs。确认 100k teacher hash；不要将 full.pth hash 当作 teacher hash。现场记录均标 UNVERIFIED，直至实际采集完成。
2. 补齐 D1 的三固定 seeds：P0/B100k/B150k 复用历史 gated Primus 协议，报告逐 volume/source 的配对差异；已有单 seed 可以保留。开发验证反复观察许可，最终测试继续冻结。
3. 首个新训练对照选择 **P0 vs B100k 的 Frozen Linear Probe**，再与匹配 gated Primus 结果一起分析 decoder 是否掩盖可读出差异。若加 ungated Primus，三模型必须均使用新 head，不能只对 B 移除 gate。
4. D3 用实际可用 checkpoints，限定 train/dev 标签，记录同实例、跨边界、空间距离、目标大小、来源和 spacing。当前不足以断定 150k 已退化。
5. D4 首先做未训练 HRA 的 B100k 双轴诊断，必要时以 P0/150k 为对照。原生 512 视野与256插值独立登记；吞吐/显存后续只在物理 GPU1/2/3 上审计。
6. 只有目标相关退化及匹配对照支持，才提出 C10/G10；只有分辨率证据支持，才提出 HR 对照。当前不启动 Gram/HRA/LAST/TopoMask，也不终止任何进程。

## 本地可复验

```powershell
python artifacts/representation_diagnosis_audit_20261013/build_audit.py
python artifacts/representation_diagnosis_audit_20261013/tools/source_smoke.py
python -m unittest discover -s artifacts/representation_diagnosis_audit_20261013/tests -v
```

CPU torch 仅安装到本交付目录 vendor，网站环境未修改。服务器只读采集示例见 [README](README.md)。提供可用登录会话后，尚未完成的真实权重诊断可继续执行；本报告没有将这些待执行项目标为完成。
