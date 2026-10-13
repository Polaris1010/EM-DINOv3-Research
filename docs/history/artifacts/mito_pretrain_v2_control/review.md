# MitoVerse LAST × TopoCL v2：执行前审查

审查日期：2026-09-26。依据用户提供的 v2 方案、本地归档、服务器实时检查及官方公开资料。**已完成预检、数据准备、拓扑校准、四配方短测和稳定性/恢复门槛；正式 seed0 SSL 队列已启动。**

## 结论

方案可以作为有条件的开发协议执行。四组比较、锁定数据和预算、独立 SSL seed、2D 主终点及资源保护的方向一致。正式 100k 训练必须等服务器实时核验、代码实现与四组 smoke 全部通过；不能仅凭本地归档认定这些条件满足。

## 已核对的本地证据

- `artifacts/mitoverse_round6/data_manifest_v3.csv` 有 212 个体积：train 31、validation 8、legacy_dev_test 16、excluded 157。train 来源为 han24 5、kunduri22 3、muller21 7、turner22 16。该文件是本地归档，服务器实时版本仍需核对。
- `artifacts/mitoverse_round6/split_audit.md` 说明原 specimen/scan/ROI 映射不完整，部分重复关系未知。v2 明确接受这一开发协议，但结果必须保留 `uncertain` 曝露，不得称为严格独立测试。
- 第五轮 `artifacts/gram_long_round5/r4_common.py` 中 N2 的真实公式是 `x.float32 / 255`、灰度复制三通道，再按 ImageNet mean `(0.485,0.456,0.406)` 和 std `(0.229,0.224,0.225)` 分通道标准化。无须因缺少公式退回 N3；服务器版本须复核。
- `artifacts/mitoverse_round6/mitonet_comparison_protocol.json` 已锁定官方 MitoNet_v1 权重、输入尺度 1、512 tile/256 stride、mask 有效区及 hash。`mitonet_comparison_summary.json` 显示 8 个 validation 与 16 个 legacy_dev_test 体积均有 MitoNet 结果。新实验可引用这个固定对照并核对代码/数据 hash；只有协议差异才重算，不为新方法挑新尺度。
- 第六轮 `stage_status.json` 的旧 `exploratory_masked_baseline` 标为 running；这是历史文件，不能推断现在仍在运行，也不能推断已停止。服务器上要查 PID、锁、协调器和 GPU UUID。

## 实施门槛和修订

1. **访问与进程门槛。** 已通过 root 身份完成服务器预检。GPU 1/2 的 PID 1971241 是旧白名单中的网站进程，各占约 458 MiB；GPU 3 空闲。GPU 0 有其他进程，不使用。网站本地 HTTP 连续三次 200，MemAvailable 约 479 GiB，数据盘可用约 268 TiB。旧第六轮协调器登记 PID 已不存活；旧阶段状态已完成。正式任务仍要由本轮协调器实时监护，不能仅凭这一时点的快照派发。
2. **数据门槛。** 复用归档 split，逐体积校验真实 Zarr 路径、img/mito/valid mask 轴序与坐标、文件集合互斥和已知别名。han24 图像可用于合格 train SSL；mask 语义没有权威证明时，五个 han24 体积统一排除监督训练。不能根据标签前景筛 SSL 图像。
3. **权重与模型门槛。** 原始自然 DINOv3 ViT-L/16 权重 SHA256 为 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`。实测有 4 个 storage tokens、启用 `mask_k_bias`；按原生架构严格加载时 missing/unexpected keys 均为 0。原生 `get_intermediate_layers` 第 6/12/18/24 层 patch 网格正确，最后一层与官方前向 patch 特征差异为 0。旧配置 `n_storage_tokens=0`、`mask_k_bias=false`，不能直接复用。旧 P1 是不同架构/协议，不可作为本轮匹配对照。DINOv3 权重受其专用许可约束，数据集许可按体积来源记录。
4. **方法门槛。** LAST 官方 commit `cdeb884af65e7774f2da80f666d95cf09a76b717` 的示例沿特征维 `dim=-1` 做 FFT、在 patch 维 `dim=1` 做 `topk(k=1,largest=True)`、逐通道 gather 后聚合。迁移到 ViT-L 时把高斯核宽度从 768 改为 1024，名称固定为 LAST-inspired。独立 CPU 合成张量对照差异 0，梯度有限且非零；接入完整 student/EMA 训练图仍待核验。TopoCL 官方 commit `a7346beca8b13efbc5abf9139fdc50158082c8bd`。v2 仅为拓扑校准增强，原论文还包含拓扑编码器和 MoE，不能称完整复现。服务器无外网 DNS；GUDHI 3.13.0 官方 wheel 已离线装入本轮独立目录，未改系统环境。
5. **算力门槛。** 四组统一 batch 的最高可行值只能由 B/L/T/LT 的 10 步 smoke、100 步稳定检查、完整恢复检查确定。未测 GPU 步时、不提供完工天数。正式任务不因 10k/30k 分数变化提前淘汰。
6. **结果解释。** W 是同一 validation 上的获胜候选；`W-B>=0.005` 是描述性进入下一阶段门槛，不是统计显著性。两个 validation 来源和 3 个 head seeds 不支持独立总体推断。legacy_dev_test 已在历史工作中查看，只能按开发测试报告。
7. **拓扑校准和短测。** 仅以 train 图像完成每来源 32 个 crop、共 128 个 crop × 12 个候选的 PH 校准；H0/H1 用 64×64、持久度阈值 0.02 和相对 bottleneck 距离。`blur_03` 与 `blur_06` 在校准尺度上近乎恒等，在任何模型训练前依照外观 MAE<0.001 的技术规则排除，原始分数保留。锁定 weak=`noise_001/002/003`，strong=`blur_10/gamma_mild/contrast_blur`。B/L/T/LT 均在物理 GPU3 上通过受守护的 batch16、10 步短测，峰值 CUDA 分配约 9.6 GiB，样本链一致，网站健康。T 首次尝试在训练开始前因共享环境缺少 `scikit-image` 失败；已把训练所需外观变换与离线校准依赖拆开，失败日志和 STOP 记录保留，安全核查后重试成功。四组 10 步损失都显示 16.625，因此仍必须审查损失分项、特征方差和参数变化的 100 步门槛，不以短测通过等同于有效学习。

## 官方依据

- LAST-ViT: https://github.com/ChengShiest/LAST-ViT
- TopoCL: https://arxiv.org/html/2603.14647v1
- DINOv3 与许可: https://github.com/facebookresearch/dinov3 、 https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- MitoVerse 数据卡: https://huggingface.co/datasets/pytc/MitoVerse

## 当前状态

服务器本轮目录：`/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227`。首批预检与数据准备完成，生成 SSL 31、监督 26、验证 8、开发测试 16 份清单和 hash；源码、环境、权重证据已归档。第二批 B/L/T/LT 的 batch16 10 步、100 步和 5+5 恢复检查完成。恢复的样本链、配置和步数完全一致；权重数值差异约 1–2×10⁻⁵，低于 B 同机两次连续运行的最大差异，**不是位级一致**。100 步 BF16 总损失均显示 16.625，分项有限、梯度非零、特征方差与参数更新通过预设硬门槛；1000 步正式作业读数约 16.5。正式准入收据为服务器 `evidence/formal_seed0_admission.json`。`mito-pretrain-v2-seed0.service` 已运行，物理 GPU1=B、GPU2=L、GPU3=T；LT 排队等待第一张卡释放。首个 1000 步完整 checkpoint 及 SHA256 收据已在 B/L/T 三组产生。网站本地 HTTP 200，GPU0 未分配。

下游数据读取、N2/有效区、2D/2.5D 中心切片配对、原生 ViT-L special-token 剔除及统一 44,576,514 参数 decoder 已通过 CPU 探针；GPU 训练和完整体积评估尚待上游权重产生后的 10 步 smoke。`mito-pretrain-v2-watcher.service` 负责依赖驱动的后续批次，遇资源、数据、数值或评估错误立即停在相应阶段，不自动覆盖失败证据。`batches.json`、`stage_status.json` 记录批次依赖；实时状态以服务器阶段文件和各 job 的 `progress.json` 为准。执行与接管细节见 `runbook.md`。**队列在运行，尚无正式最终分数或研究结论。**
