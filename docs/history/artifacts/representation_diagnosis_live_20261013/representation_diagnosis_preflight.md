# Representation Diagnosis：实时核验与下一阶段准备

日期：2026-10-13。目标：为电镜通用线粒体分割、最终与 MitoNet 的公平比较建立可核验基线。**本轮核验和限定 smoke 已完成；完整 D1–D4 性能实验尚未完成。**

## 1. 核验后的研究基线

1. P0 / B100k / B150k 已完成 CMCC、CMCT 的共同冻结协议单 seed 评价；不是“P0 尚未运行”。B100k 在这两个开发来源优于 P0，150k 未取得一致的进一步提升。
2. 当前 head 是含四个启用频域门控的 modified Primus。三模型初始 head 参数哈希相同，encoder 真正冻结。历史同 decoder 比较有效，但不能称为未经修改的官方 MedDINOv3 复现。
3. B100k / B150k 均为 vanilla SSL，Gram=false、global/local=256/112，没有独立 HRA。本轮完整扫描实际训练日志并读取主 checkpoint metadata。
4. CMCC / CMCT target 是 label==1；既有三指标重算是 mitochondria-only。验证体积数量分别为 3 / 2，历史 SD 是体积间 SD；距离单位尚非实际 nm。
5. B_full 是全 MitoVerse 的 transductive 开发资产。正式 manifest 已取得，但 acquisition 隔离、标注完整性、有效 mask 等未因此自动解决。B_clean 的现有 NO_GO 保留。

详见 [Decoder audit](decoder_architecture_audit.md)、[Gram/HRA audit](dinov3_stage_completion_audit.md) 和 evidence 目录的服务器原件。

## 2. 已实现和实际 smoke

| 工作 | 实现 | 本轮真实测试 | 尚未完成 |
|---|---|---|---|
| 共同 Primus | strict DINO loader；实际 server Primus；独立 seed 初始化 | 三模型真实加载，head 初始化 hash 一致；各 2 steps head 优化 | 新三 seeds 完整评价 |
| Frozen Linear Probe | 最后一层 1×1 Conv；可登记四层拼接；valid-mask CE+Dice | 三模型各 2 steps，loss / gradients 有限，encoder 无更新 | 完整 train/dev readout 指标 |
| Dense features | 四层 pure valid patch；fg/bg centroid、方差、有效秩、邻接 cosine 分层 | 相同 CMCC 训练样本上三模型均输出四层诊断 | 多 ROI / 多 source / 时间点；完整边界与实例测量 |
| 分辨率 | 256 / 384 / 512；fixed_fov 与 native_fov 两种模式 | 三模型×两模式×三尺寸，共 18 次真实 EM forward | 已训练 head 的配对 Dice / HD95 / NSD、统一输出网格 |
| Paired 输入 | 同 source、同原生 ROI、input hash、crop 坐标和 declared spacing | 各 mode/size 输入 hash 在三模型间一致 | acquisition 隔离证明、锁定正式物理评价 ROI |
| 暴露记录 | 精确正式 SSL manifest；CMCC/CMCT 每个 train/val case×模型记录 | 162 行 readout case/model 暴露记录；ROI metadata 与 input hash | 逐原图上游自然训练重叠、实际 SSL 抽样命中、最终测试治理 |

代码：[diagnosis.py](scripts/diagnosis.py)、[server_factory.py](scripts/server_factory.py)、[live_smoke.py](scripts/live_smoke.py)。复用此前预审实现，在独立目录部署；没有修改历史训练或评价脚本。

FrozenReadout 强制 encoder.eval() 与 no_grad/detach；参数文件 strict load。Linear 默认最后一层 1024→2，仅 2,050 参数。PrimusHead 保留实际 gates。fit_readout 仅优化 head，限制 1–3,000 steps，拒绝带 encoder autograd 的缓存。

fit_readout 是新的 cached paired-ROI 诊断协议，不等同于历史 nnU-Net sampling / sliding / TTA；其指标不得混入历史六组结果。多 seeds 重跑历史 Primus 应复用已审计的 label1 probe 和训练流程。

## 3. 真实 smoke 输入与解释

### Head / dense

共同输入为 CMCC_crop20 的 fold-0 **训练**样本，224×224；标签 -1/0/1。只在标签非负位置计算 loss。三模型各训练两个 head 各两步，总计 12 个 head updates，未保存或修改任何历史 checkpoint。每个模型结束后，encoder 全部参数仍 requires_grad=False / grad=None，完整 state_dict 哈希不变。

当前 dense 样本有 28 个 pure valid patches，其中前景 10、背景 18；四层指标正常计算。**pure-filter 后跨边界邻接 pair 数为 0，delta_boundary 为 null。** 不能把 null 当作 0 或据此宣称边界退化。下一轮需登记足够多的开发 ROI，并审查是否需要另设基于真实 GT crossing 的 mixed-patch 边界指标；改变指标时另命名，不能修改 gate 来强行通过。

CMCC 的二值 mask 只能支持同前景类别的一致性；没有实例 ID 时不能称“同一线粒体内部”。有效秩采用 singular-value entropy，单一小 ROI 的秩或 L2 不能证明 collapse / dense degradation。

### 分辨率 / 物理 ROI

真实图像：MitoVerse `kunduri22_train01` 的 img，formal legacy_split=train；z=192，y/x=[256,768)，原生 512×512 uint8。选择依据为 train 身份与尺寸，没有用测试 GT 选择位置。此 smoke **只读取图像，不使用该卷 GT 或假定有效 mask**。

源 metadata 声明 isotropic 16 nm，512 ROI 对应 8192×8192 nm；各轴等距，避免将未知 anisotropic 顺序强行转换。物理数值仍标 declared metadata，不冒充重新校准；原始 acquisition 独立性 UNVERIFIED。

| 模式 | 256 | 384 | 512 | 控制关系 |
|---|---|---|---|---|
| fixed_fov | 512 原生 ROI 重采样 | 同 ROI 重采样 | 原生 ROI | 视野相同，输入采样密度不同；插值不创造信息 |
| native_fov | 原生中心 256 crop | 原生中心 384 crop | 原生 512 crop | 原生采样率相同，视野随尺寸增加 |

两种模式中，同一 size 的三模型输入完全一致；native_fov 不保证不同 size 的物理视野一致。token 数分别 256 / 576 / 1024，logits 分别 1×2×size×size，均有限。

分辨率 smoke 用固定 raw uint8/255 与 ImageNet RGB normalization；head smoke 用历史 nnU-Net 预处理图像。这是两个明确不同的技术测试，未混算性能。分辨率 forward 使用经过仅两步技术优化的临时 head，输出有限不说明其分割精度。

## 4. 算力、保存与测试证据

[real_weight_smoke.json](smoke/real_weight_smoke.json)：三模型 PASS。GPU3 上本轮单模型 CUDA peak allocated 约 2,725 MiB，模型 smoke 时段约 8.42 / 4.85 / 4.80 秒，另有读文件 / hash 时间。这是冻结 batch-1 forward/head 更新的开销，**不能作为 HRA encoder backward 的显存或吞吐估计**。

[immutable_inputs_check.json](smoke/immutable_inputs_check.json)：三份主 encoder checkpoint、正式 manifest、既有 metrics、训练/评测源码和配置的前后 hash 完全一致。11 份 teacher exports 的实际 hash 与 metadata 已另行核验。

只在 physical GPU3 使用 CUDA。GPU0 未使用；网站与未知进程未终止。结束后 GPU3 回到 0 MiB，网站 HTTP=200，其他卡原有进程继续保留，见 [final_server_state.json](evidence/final_server_state.json)。历史 Stage2A FAIL、checkpoint、manifest 和原结果未修改；没有启动任何新 SSL / Gram / HRA / LAST / TopoMask 长训。

## 5. 未核验项与正式执行限制

- acquisition ID / ROI 来源隔离：UNVERIFIED；目前只确认 case ID train/val 不交叉。
- annotation completeness / Han24 mask 极性 / 原始有效区：UNVERIFIED；不能把未知背景当作正式诊断 GT。
- CMCC/CMCT 实际 nm spacing 和 raw↔preprocessed 几何：UNVERIFIED。
- 官方自然权重逐图上游 corpus overlap：UNVERIFIED；“没有 EM 续训”不证明任何图像都未见过。
- 完整三 seeds、Linear benchmark、跨来源 D3、配对分割 D4、full fine-tuning 和 MitoNet 同协议对照：未运行。

[diagnosis_plan.unlocked.json](evidence/diagnosis_plan.unlocked.json) 已补齐三 checkpoint 实测 hash，但仍为 UNVERIFIED_ROIS_NOT_LOCKED，rois=[]。正式 feature runner 会拒绝此计划；没有填造 acquisition / annotation 信息使其通过。此次独立 smoke 不冒充 locked 实验。

## 6. 下一轮顺序与判断规则

1. **先固定数据协议。** 沿用 Stage0 source/acquisition 规则，核实 valid mask 与物理空间。B_full 可保留用于开发，B_clean 仅在 clean manifest 被证据解除 NO_GO 后开始；本轮不训练 B_clean。
2. **D1 / D2 优先。** P0、B100k、B150k 使用相同历史 gated Primus 补齐三 seeds；优先完成 P0 vs B100k 的 Frozen Linear 对照。若加入 ungated Primus，P0/B 使用同样的新 head 与预算，单独报告 gates 效应。先区分可读出性与强 head 的影响。
3. **D3 取实际可用时间点。** 10k/25k/50k/75k/100k/125k/150k，在训练/开发数据上记录方向、邻接距离、目标大小、source、spacing、实例标注和 boundary 有效 pair 数。不要用最终测试标签选 Gram reference，不单凭秩下降或现有单 seed Dice 下滑认定退化。
4. **D4 做分割尺度对照。** 已训练、冻结的共同 head 预测还原到同一原生目标网格，再计算配对 Dice / HD95 / NSD。fixed_fov 与 native_fov 分开；控制 normalization、滑窗、TTA、valid mask、spacing。跨 FoV 时使用共同中心评价 ROI，避免评价对象随输入改变。
5. **证据后再提 Gram / HRA pilot。** 支持 dense 退化才做 C10/G10；支持真实输入尺度收益或尺度失配才做 HR 对照。各组保持同起点，记录 updates、图像暴露、GPU-hours、FLOPs，加入无 Gram HR。先审计 encoder backward 的显存与有效 batch。
6. **MitoNet 为最终任务基准。** 用锁定的同图像、同线粒体 target、有效标注区与 split 做公平比较；目前两来源 frozen 开发结果不能证明已超过 MitoNet。

当前最有价值的下一项新 readout 实验是 **P0 vs B100k Frozen Linear**，同时补齐 gated Primus seeds；当前最有价值的新预训练对照仍需 D3/D4 证据后决定。P 继续暂缓，crop 支线保持探索状态。
