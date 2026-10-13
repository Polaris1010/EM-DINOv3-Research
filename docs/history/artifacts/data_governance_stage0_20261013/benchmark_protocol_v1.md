# DINOv3 × MitoVerse — Protocol R/S v1

审计日期：2026-10-13。执行范围：Stage 0。**协议规则固定；数据成员与正式运行尚未锁定，当前 NO_GO。**

## 数据身份和隔离

数据来自统一 registry。asset_id 是资产索引，不是虚构的 acquisition 身份。`acquisition_key_candidate` 仅用于追查同名来源；只有原始来源、采集身份、ROI 坐标与坐标变换经证据核验后才能填写正式 acquisition_id。

重复关系：L0 为文件或解码像素完全相同；L1 为同一物理 ROI 的裁剪、配准或灰度表示；L2 为同 acquisition 的不同区域；L3 为同源实验或域。pHash 只用于候选检索，不能确证 acquisition。不同 hash 不能确证独立。当前 CSV 中带 candidate 的关系未通过像素验证。

严格 unseen-acquisition 主协议：保留测试 acquisition 的全部资产从 SSL 和监督训练排除，包括 MitoVerse、CEM、IsoOrg 的其他 crop。dev 也与 train 按 acquisition 隔离。存在身份不明或待解释冲突的资产进入 quarantine。禁止恢复旧 31-volume 白名单。

官方 benchmark 的同 acquisition 不同 ROI 另设 region-generalization 结果表。保留官方角色，但必须核验区域互不重叠和原始坐标变换。该表不能充当 unseen-acquisition 证据。两个不同 crop store 的本地坐标不能直接比较；没有 parent-space 映射时 loader 拒绝准入。

ROI 为 ZYX 半开区间，明确 stored-array 坐标系、parent acquisition 坐标系和两者的变换；统一 nm 单位。所有 view、resize 和增强只能在准入区域内执行。Wei20/ME2-Pyra 未获得实际区域之前禁止按完整 volume 标 train。CellMap 当前 overlay 的暂定 train 不具备正式 split 的效力。

CMCC、CMCT、历史验证及 legacy_dev_test、424 图 PH/crop/GT 诊断涉及区域保留 method_development_exposed。开发曝光不等于监督训练标签泄露，但已反复用于选方法的区域不能重新命名为 untouched final test。旧模型对真正外部、未暴露 acquisition 的评测资格逐来源判断。

## 标签与有效评价区域

MitoVerse `mito > 0` 是候选语义前景规则，实例 ID 的体积连续性和标注完整性另审计。CEM-MitoLab 的官方声明是每幅图像内 1–255 的实例 ID；不能拼接成跨切片 3D 实例。mask 极性必须由官方转换过程或标注说明证实，不能根据前景与 mask 的相关性猜测。

loss 和全部指标仅在 valid annotation region 内计算。未经核验的 0 不默认可靠背景。部分标注只在显式证明可评价的有效区域内使用；边界截断实例须按官方协议处理。使用 ignore mask 后，距离指标不能把人为裁切的 ignore 边界当物理表面。

## 预训练实验

P0：官方自然图像 DINOv3 ViT-L/16，历史权重 SHA256 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`（本轮未重新读取模型文件）。所有干净实验从这一权重开始，不从 B_full 开始。

先建立 A1/M0。数据轴 A1–A4 与方法轴 M0–M4 分离。每项正式预训练均为 100,000 optimizer updates、global batch 48、3 卡各 16、global crop 256、8 个 local crop 112、BF16；vanilla 损失 DINO=1、iBOT=1、KoLeo=0.1。正式配置还必须从实际成功训练的 resolved config 固定学习率、调度、优化器、增强和 teacher/student 导出方式；禁止复制已被取代的单卡预备配置。初始化、manifest、代码、优化器配置和预算逐项记录 SHA256。

数据实验固定优化器步数、batch、crop、增强和解码器；短程筛选的 backbone 与预算在结果产生前固定。记录每来源实际曝光次数、独立采集体数量、可追溯 ROI 数和模态/像素尺度/组织/物种/制样定义的域组成。CEM 的 651 文件夹不自动等于 651 独立 acquisition。不得把图像数、采样权重和方法一起改变。A5 的采样权重消融单独注册。

B_allMV_transductive_100k/150k 保留为 transductive 历史资产，报告改名不修改 checkpoint。100k→150k 是续训，且实际为 100k 后固定 minimum LR 1e-6；不能直接归因为独立初始化下的纯训练长度收益。旧 L/LT 不列为新论文正式结果。TopoCrop 保留方法开发负面证据，不启动长训。

## Protocol R：表征质量

比较 P0、B_clean、干净的新数据控制组、Our SSL。历史 B_allMV 单列图像暴露诊断。固定训练标签、解码器实现、初始权重、训练 budget 和增强，主冻结 encoder probe 与 end-to-end fine-tuning 分表。

主 probe 候选沿用已存在的 Primus decoder：encoder frozen，BF16 特征、FP32 decoder，batch 1，CE+Dice，3,000 decoder updates，50% 前景和 50% uniform sampling。这只是预注册候选；未获得完整成功配置及可跨来源执行的训练样本映射前不宣称运行锁定。正式 decoder seed 为 20260907、20260908、20260909；同一 seed 的各模型使用相同标签子集、抽样序列与初始 decoder。使用 dev 选择 checkpoint、阈值和后处理。最终测试只运行预先固定的模型和代码。

少标签预算候选为 1%、5%、10%、100% 的可评价训练标注区域；必须在 train acquisition 内抽样，并锁定同一子集。ROI 取整、最少实例数和抽样随机种子在正式运行 lock 中记录，不能以 test 表现选子集。独立 ROI/实例的标注预算另报，不能以相邻切片伪装独立标注。

每 volume 报 Dice、Precision、Recall、NSD、HD95；来源内 volume 等权再做 source macro，同时报 worst-source Dice。距离主结果使用核验的物理 nm spacing；NSD 容差仅由 dev 和任务要求在测试前选定并写入运行 lock。历史 spacing=(1,1,1) 数值明确标 voxel units，不转换成物理结论。形态指标与 3D 实例 AP 分表。

空集政策：双方全空 Dice=1；GT 非空而预测为空 Dice=0、Recall=0、HD95=+∞、NSD=0；GT 空而预测非空 Dice=0、Precision=0、HD95=+∞、NSD=0。无正例时 Recall 记 NA，无预测时 Precision 记 NA；报告空体积数量与无穷距离失败率，有限 HD95 均值不能掩盖失败。测试前固定并验证 evaluator 实现、连通性、边界与 ignore 处理。

报告 3 seed 的配对差异。以独立 acquisition 为 cluster，保留同 cluster 内全部 volume 与模型配对，bootstrap 10,000 次，固定 seed 20261013，报告 95% CI。seed 不能伪装独立 acquisition；测试域不足时如实报告数量和不确定性，不能按切片 bootstrap 得到过窄区间。

## Protocol S：完整分割系统

独立报告 official MitoNet zero-shot、可行时统一标注数据微调的 MitoNet、DINOv3 固定分割系统、Our SSL 同一系统、nnU-Net 2D 与可复现 EM/OrgMIM 基线。

二维语义、二维实例、三维实例三类任务分表。MitoNet semantic 结果由正式固定的实例前景转换得到，不能与其他模型 3D 实例成绩混表。固定 tiling、spacing/resampling、TTA、阈值、连接和后处理，所有参数用 dev 选择。为官方 checkpoint 与微调后的 checkpoint 分别建立训练来源/标签/方法开发曝光表。

MitoNet 的训练来源排除是原论文的声明，不能替代本项目资产审计。任何统一微调需要核验 CEM-MitoLab 与最终测试的 acquisition 关系。零样本系统与统一微调系统各自比较；不能仅用标签量不同的更高 Dice 声称系统更强。

实例指标遵守对应 benchmark 官方规则，另报困难实例的 merge/split 和召回率。困难实例大小、接触、弱边界分层定义仅在 train/dev 建立。OrgMIM 是必须比较和区分的候选强基线；其预训练数据和 SAM 先验也须记录曝光条件。[OrgMIM 官方仓库](https://github.com/yanchaoz/OrgMIM)。

## 最终冻结与启动规则

Data lock 必须包含 acquisition/ROI 身份证据、明确 train/dev/test/external 成员、有效标注区域、跨数据重复审计、独立外部测试与排除记录。Run lock 必须包含完整训练/evaluator resolved config、模型/代码/数据 SHA256、decoder seeds、NSD 容差、标签子集、后处理和终止规则。首次正式 test 之前锁定最终候选。任何变更提升版本号并记录原因；test 不能选预训练机制。

Stage 0 交付和 go/no-go 判断不包含训练授权。当前清单 `training_permitted=false`、`entries=[]`；loader 必须拒绝。**本轮不自动启动 B_clean。**
