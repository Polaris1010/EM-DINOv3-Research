# MitoTopo-DINO 第一轮审计与短测报告

日期：2026-09-08（香港时间）。远端：192.168.203.97，mira-4090-7。
远端交付目录：`/opt/data/3_117/tanghr/codex/mitotopo_20260908_first_round/`。

**结论：已完成仓库事实盘点、数据元数据审计、CPU 严格加载、5 个 patch 的 EM 主干接入推理及部分单卡短测。短测计划 200 step，日志记录至第 161 step，因系统 swap 持续增长被资源保护器停止；不标记通过，不进入长训。**

本报告把事实、推断和未完成项分开记录。没有将随机初始化分割头的指标当作已训练模型迁移能力，也没有生成不存在的最终训练 checkpoint。

## 1. 交付状态

| 要求 | 本轮结果 |
|---|---|
| 两仓库审计、Git 状态、模型链路 | 已完成主要链路；原始 dirty files 保留，源文件快照及 diff 已保存 |
| checkpoint 清单 | 180 个文件：42 个 EM teacher 导出、64 个下游权重、2 个 encoder 字典、72 个分布式分片；108 个非分片文件已读取 keys，无读取错误 |
| SHA256 | 本轮选用的 EM、自然初始化、下游候选三个权重已计算；其余未计算，不冒充完整哈希审计 |
| 数据清单 | 212 卷全部存在，13 个来源；读取元数据，未全量扫描像素 |
| 抽样可视化 | 固定种子抽查 4 卷、12 个 patch；保留较早审计预览及配对增强图 |
| CPU 样本与模型 | 双 worker DataLoader、确定性样本、严格 encoder 加载、CPU 前向和小张量 loss/backward 通过 |
| EM checkpoint GPU 推理 | GPU 3，5 个 192×192 patch，无 TTA，保存输入/标签/预测/概率图 |
| 50–200 step 短测 | 至少执行并记录到 161 step，验证到 150 step；资源停止，未完成计划 200 step |
| 训练权重保存及重载 | 未到达执行点，因此未验证；不能把 CPU 的历史权重 key 匹配代替此项 |
| 资源日志 | 已保存 5 秒间隔 GPU/RSS/系统可用内存/swap/I/O wait，以及两次 preflight |
| 正式 E0 命令与耗时 | 暂未具备给出“已验证正式全量命令”的条件；可执行的受保护短测复查命令见第 9 节 |

主要文件：[数据清单](dataset_manifest.csv)、[指标摘要](metrics.json)、[实验登记](experiments.csv)、[资源记录](resource_log.csv)。
详细原始输出、源代码快照、环境清单和逐 patch 指标随交付包提供；早期本地同步副本位于 `remote/`。

## 2. 真实仓库与环境

文档给出的两个路径是父目录，实际 Git 根目录是：

- `/opt/data/3_117/tanghr/DINOv3/dinov3`：main，commit `91ffb6803c74aeb6927209b82560142743763dd7`。
- `/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3`：main，commit `bd6a650dda4e32d22e913cfc28458df24bb39b83`。

第一个仓库已有 `run_med.py`、`local_dino/` 和可视化脚本等未跟踪文件；第二个仓库已有 10 个 tracked 文件修改，包括 `dinov3Trainer.py`、`primus.py`、归一化和 DataLoader，以及多项未跟踪数据/脚本。开始与结束的 `git status` 一致。本轮没有覆盖权重、配置、结果或原数据；新增代码全部放在独立交付目录。

| 用途 | Python/环境 | PyTorch | torch CUDA / cuDNN |
|---|---|---|---|
| 现存预训练环境 | 3.11.14，`/opt/data/3_117/tanghr/dinov3_shared`，conda | 2.10.0+cu128 | 12.8 / 91002 |
| 本轮下游与 CPU/GPU 短测 | 3.11.9，`/opt/data/3_117/tanghr/envs/meddinov3`，conda | 2.5.1 | 12.4 / 90100 |
| 独立 MitoVerse 环境（仅探测） | 3.11.15，`envs/mitoverse` | 未核验 | Zarr 2.18.7 |

本轮实际使用的 meddinov3 环境：NumPy 2.4.4、torchvision 0.20.1、nnunetv2 2.5.2、Zarr 3.1.6、psutil 5.9.0、tifffile 2025.6.11。两套 `pip freeze` 已保存。

服务器驱动为 535.171.04，`nvidia-smi` 显示 CUDA 12.2。该显示值与 Python 包所带 CUDA runtime 版本并不相同；本轮下游 GPU 运算已实际启动，但没有据此声称预训练环境当前 CUDA 12.8 也已兼容验证。现存环境版本也不等同于三月历史训练时的环境快照。

## 3. 权重血缘与模型结构

```text
自然初始化 vitl16 (LVD1689M)
  ↓ run_med.py → dinov3.train.train.main，SSL teacher/student
EM med_checkpoints_continual_v2 / training_124999 / teacher_checkpoint.pth
  ↓ 选择 teacher.backbone.*，去掉 backbone.，严格加载
vit_large：patch16，1024通道，24层，0个 storage token，mask_k_bias=False
  ↓ 输入灰度按 (x/255-0.485)/0.229 归一化，复制3通道
get_intermediate_layers(n=[5,11,17,23], reshape=True)
  ↓ 每层独立 LaSt_FrequencyGate
4×1024 通道拼接 → PatchDecode → 2类 logits
```

真实训练类是 `meddinov3_base_primus_multiscale_Trainer`，位于：
`nnUNet/nnunetv2/training/nnUNetTrainer/dinov3Trainer.py`。
模型类 `Primus_Multiscale`、`LaSt_FrequencyGate`、`PatchDecode` 位于该目录下 `dinov3/dinov3/models/primus.py`。

- `[5,11,17,23]` 是 **从 0 开始的索引**，即通常从 1 计数的第 6/12/18/24 个 block，不应与“第 5/11/17/23 层”混淆。
- backbone 输出去掉 CLS/storage tokens，经归一化后 reshape 为 BCHW。本轮每层为 1024×12×12。
- gate 将 BCHW 展平为 BNC，在 **C 维**执行 FFT/fftshift、高斯平滑、逆 FFT；残差标准化后由卷积 projector 生成门控。不是沿 H/W 的空间频谱增强。
- gate 的 sigma/gamma 可学习；gamma 初值 1e-3、限制在 [0,0.1]。原实现包含 `nan_to_num` 与 clamp，不能仅凭输出有限就证明全部中间表示从未异常。
- 拼接 4096 通道后，4 次 stride-2 ConvTranspose2d，依次 2048/1024/512/256 通道，最后 1×1 卷积输出 2 类。
- 实际参数量 349,841,162；本轮全部可训练，未冻结主干，未使用多卡。

### 3.1 EM 起点证据

`med_checkpoints_continual_v2/config.yaml` 明确 `gram.use_loss=false`、`gram.compute_stats=false`。历史日志记录至 iteration 124999，总 loss 12.483235，local/global batch 为 16/64，可见历史四进程记录。本轮不重新预训练。

当前下游代码硬编码的是 `/mnt/mira_datacenter/.../med_checkpoints_continual_v2/...`，不是本服务器文档中的 `/opt/data/3_117/...`。独立适配器修正了路径而未修改原类。

严格 CPU 核验：missing=[]、unexpected=[]、shape_mismatch={}。仅排除明确属于预训练的 DINO/iBOT heads；逐键清单见 `em_key_match.json`。不存在位置编码插值或以 `strict=False` 静默容错的操作。

自然权重相对该 EM backbone 多出 `storage_tokens` 和 24 个 `attn.qkv.bias_mask`，共有参数的形状一致。因此未来 E0a/E0b 需要显式控制这些结构差异，不能只交换文件路径便称为纯初始化消融。v3/v4 的配置也不同，本轮未混用。

### 3.2 选用文件的 SHA256

| 类型 | 文件 | SHA256 |
|---|---|---|
| E0b 主干 | `DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth` | `53f759c7019f8753d9d18b52aa79168b51b7480466934df47f446bc3d5f696524` |
| 自然初始化候选 | `DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55ff6ec4f1964d8890f66035` |
| 历史分割候选 | `Med_DINOv3/nnUNet_results/Dataset_13799_CMCT_upgrade2_v4/meddinov3_base_primus_multiscale_Trainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth` | `6424ee05c840cd520ff44ec76bc4bee98d28a6481839faa34e04d60a67884ea9e` |

历史分割候选也无 key/shape 不匹配，但其内嵌 dataset_name 是 `Dataset131_CMCT`，current_epoch=1，_best_ema≈0.327816。目录名与内嵌名称不同，不能仅凭 `checkpoint_best` 文件名确定它是所有实验中的最佳模型。该权重没有用于本轮训练，原计划的独立 GPU 推理未执行到。

72 个 `.distcp` 仅盘点路径/大小与关联配置，尚未通过分布式 checkpoint loader 恢复。清单没有将这些分片冒充可独立加载的 teacher。

## 4. 原训练与推理链路

原下游以 nnUNet plans/configuration/dataset_json 构建模型，通过 `nnUNetv2_train`（`nnunetv2/run/run_training.py`）选择 trainer，`-num_gpus` 控制单卡/DDP。原 EM trainer 默认 batch=4、每 epoch 100 次训练/20 次验证、300 epochs、5 epoch warmup；AdamW betas=(0.9,0.98)，主干目标 LR 3e-6、其余 3e-5，weight_decay=0.05。前 100 step 的实际 warmup LR 为 1e-7/1e-6，第二个 epoch 为 6.8e-7/6.8e-6。

原数据实现有 NPY mmap、压缩 NPZ 和 blosc2 路径；压缩 NPZ 路径会展开 case，不适合直接承接大型 Zarr volume。原默认交叉验证是 case 级 nnUNet split，不自带 specimen 关系，不能据此证明相邻 ROI 无泄漏。

原推理入口 `nnunetv2/inference/predict_from_raw_data.py` 使用 nnUNetPredictor，默认 tile_step_size=0.5、Gaussian overlap blending、mirroring=True，可保存概率。本轮只做无 TTA patch 推理；没有验证整幅滑窗边缘缝隙、全卷输出格式或完整 volume 聚合指标。

本轮保留原模型类、归一化、Dice+CE loss、AdamW 参数组和 warmup；适配部分是 Zarr patch loader、192 crop、batch=1、同步翻转增强和非官方划分。它是恢复链路的 smoke test，**不是原 nnUNet 全训练策略的严格复现**。

## 5. 数据审计与划分

数据根目录：`/opt/data/3_117/tanghr/MitoVerse_data`。catalog 中 212 卷均存在，13 个来源；元数据检查未发现图像/标签/catalog shape 不一致。格式为 Zarr v2，`img` 为 uint8，`mito` 为整数实例标签，0 表示背景，语义标签为 `mito>0`。

`catalog/datasets.yaml:4` 明确 `voxel_nm=[x,y,z]`，而 `shape_zyx`/array 为 ZYX。最终清单已将 voxel_nm 反转为 ZYX并保留原值。例如 jiang25 是 (40,15,15) nm。首次审计中保留的“顺序未核验”注记是当时状态，后由这条源码证据解决；本輪不使用物理尺度变换。

常见 chunk=(64,512,512)，Blosc/zstd clevel=5。本轮样本按小 patch 读取，实际解压粒度仍是 chunk；无全卷缓存。数据审计记录的峰值 RSS 约 0.214 GiB，权重 metadata/hash 审计约 0.390 GiB。

清单中的 `num_instances` **来自 catalog，未流式重计**；全卷 foreground_ratio 留空，抽样比例在 `sample_audit.json`。不能据稀疏样本断言全卷不存在断层、漏标、空切片，或实例 ID 已跨 z 一致。许可证是 per-volume 的 `other`/来源引用形式，尚未逐源完成正式使用核验。

本地 MitoEM 2.0 overlay 共 25 条（14 train / 1 val / 10 test），与 README 表格宣称的细分计数不完全一致；Pyra 的区域划分仍是 TODO，Kunduri 标注说明 train01/train02 ID 尚未连通；Jurkat/Macro/Podo 同一 specimen 的不同 ROI 被列入不同 split。这些足以阻止直接使用为正式无泄漏基准。

用户已明确允许非官方探索划分。本轮使用：

- train：仅 `jiang25_vol1`（原 overlay 的 train 卷）。
- val：`conrad23` 全部 5 卷，固定 5 个 patch 进行验证。
- test：未设置、未使用。jiang25 原 val/test 卷不参与训练。

使用数据来源作为比 specimen 更保守的分组边界，因此本轮 train/val 无同源 specimen 交叉。但尚未排除这些数据与历史自监督预训练图像重叠。现有预训练目录明确包含 JRC Jurkat/Macrophage/Kidney 等原始来源，未来跨域结论必须同时审核预训练暴露，不能只审核下游 split。

可视化：

![训练来源样本](previews/jiang25_vol1.png)

![验证来源样本](previews/conrad23_c_elegans.png)

## 6. 实际测试结果

CPU 已验证输入 [1,192,192] float32、范围约 [-2.118,2.249]，标签 int64 0/1；同种子同索引样本及双 worker 输出一致。模型 CPU 64×64 输出 [1,2,64,64]，无 NaN/Inf；合成 loss/backward 有限。语义指标覆盖完美、全背景、全前景及 split/merge 对二值指标不可见的测试；AP/PQ/ARE/VOI 等实例指标尚未实现或验证。

GPU 推理使用已有 EM encoder + 新初始化 decoder/gates，5 个 192×192 patch，BF16、batch1、无 TTA，输出空间尺寸正确且有限。图像见 `predictions_preview/before_*.png`。该模型尚没有已训练的下游分割头，因此初始指标不是“EM 迁移效果”的正式证据。

同一组固定验证 patch：

| 阶段 | patch Dice 均值 | 汇总抽样像素 Dice | 汇总抽样像素 IoU | Recall |
|---|---:|---:|---:|---:|
| 训练前 | 0.083706 | 0.088468 | 0.046281 | 0.418433 |
| 50 step | 0.075545 | 0.079532 | 0.041413 | 0.223400 |
| 100 step | 0.062277 | 0.063703 | 0.032899 | 0.101766 |
| 150 step | 0.014029 | 0.012482 | 0.006280 | 0.007285 |

**这些均是 5 个 patch 的指标，不是 volume-level Dice。** 每个 patch 的坐标、前景比例和 precision/recall 均在对应 JSON 中。

训练日志首条 loss=0.562707，第 141 条=0.122556，第 161 条=0.309632；不同 patch 的 loss 不能当作同一样本的单调下降。固定种子初始 loss 重复计算的断言通过，但数值未在中断前落盘；训练结束后的固定 probe 尚未执行。门控梯度检查已执行，首步每个门控参数的梯度范数见 `first_gate_gradients.json`。

验证前景 recall 快速下降，提示当前极短、单训练卷跨来源测试向背景收缩；不能据此判定 EM 预训练无效，也不能作为 E1–E6 晋级依据。没有与 E0a 做公平、同架构、同训练量比较。

## 7. 资源停止与限制

- 只使用物理 GPU 3，CUDA_DEVICE_ORDER=PCI_BUS_ID，CUDA_VISIBLE_DEVICES=3。GPU 0 未使用。
- GPU 1/2 的 PID 1971241 经 cwd、命令行、用户和 executable 检查，确认为现有 MIRA uvicorn 网站服务；未干预。
- 第一次 preflight 检出 swap 连续增长，未启动训练。权重审计完成后 `vmstat 5 7` 显示 30 秒稳定，再次使用原阈值 preflight。
- 第二次通过 preflight 并启动训练；监测到 swap 1.967239→1.970413→1.973343→1.976517 GiB 连续增长，触发停止。没有为了完成测试放宽阈值。
- 执行时后台 nohup 继承 SIGINT 忽略状态，首次 SIGINT 未终止子进程；保护器 20 秒后仅对自身 Popen 创建的训练 PID 发送 SIGTERM，exit=-15。日志因此继续记录至 step161。已修复信号注册并通过 CPU 信号回归；修复后的中断权重保存尚未经实际 GPU 回归，不宣称已通过。
- 最后确认本次训练/审计/worker 均已退出，GPU 3 显存回到 0 MiB，网站进程保持运行。

5 秒采样记录的峰值：GPU 7342 MiB；主进程 RSS 2.411 GiB；包含 worker 的进程树 RSS 2.984 GiB；系统 MemAvailable 最低 475.291 GiB。它们是**采样峰值**，不能代替未落盘的 CUDA allocator 精确峰值。资源条件的失败原因是 swap 增长，不是 RAM 耗尽或 GPU OOM。

## 8. 改动与后续必须解决的问题

新增 `audit.py`、`runtime.py`、`guard.py`、`smoke.py`、`supplement.py`；原仓库无代码写入。执行版本在 `source_snapshot/*_executed.py`；当前 smoke.py 额外恢复 SIGINT handler、支持 `MITOTOPO_STEPS=50..200`，并尝试在中断时保存标为未验证的 checkpoint。修复版只通过语法/信号级检查，未重新启动 GPU。

下一步应先完成：

1. 查明短测时系统 swap 增长来源；保持原资源保护条件，再完成一个可保存/重载的短测闭环。
2. 从多个下游结果中核定 E0c 真正候选、对应代码快照和训练 split；本轮候选仅证明结构匹配。
3. 完成官方 split/区域/许可证及预训练暴露核验；对探索 split 与官方基准分别命名。
4. 完成惰性全卷滑窗、聚合指标和边缘接缝检查；不能以 patch 指标替代。
5. 为 E0a/E0b 对齐 storage token、qkv mask 等结构变量。
6. 完成流式实例统计、跨 z ID 连续性及实例指标测试后，才考虑 boundary/z-affinity。

## 9. 可复用命令与耗时边界

下面命令仅准备并运行一个**新的 50 step 复查**，带原资源门槛。它不是已经验证通过的正式全量 E0 命令，本轮未执行它：

```bash
BASE=/opt/data/3_117/tanghr/codex/mitotopo_20260908_first_round
RUN=/opt/data/3_117/tanghr/codex/mitotopo_$(date +%Y%m%d_%H%M%S)_recheck50
mkdir "$RUN"
cp "$BASE"/{runtime.py,smoke.py,guard.py,exploratory_split.json,cpu_check.json,em_key_match.json} "$RUN/"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 MITOTOPO_STEPS=50 \
  /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$RUN/guard.py"
```

本轮稳态日志中的纯计算 step 中位数约 0.136 秒；30000 step 单纯计算外推约 1.13 小时，**不包含数据等待、worker 启动、全卷验证、保存、正式 crop 和其他配置差异，因此不能作为完整 E0 的准确耗时**。50 step 复查可暂预留数分钟，但是否能够运行首先取决于 preflight。

正式 E0 完整基线的准确命令和总耗时目前仍待上述资源、划分、模型对照及全卷评测问题解决后给出。没有自动进入 E0 长训或 E1–E6。


