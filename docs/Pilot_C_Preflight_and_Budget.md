# Pilot C-1：选择性关系屏障预检与预算

C-1预检通过：完整30k训练状态可恢复，6项GPU单元测试通过，C0/C1各完成20次更新并停止。60组rank-step的实际输入、mask、RNG、采样链及日程逐项相等；C0的首20步loss还与历史D2续训逐值相同。Barrier有非零backbone梯度，固定λ=1，无需调整margin、分位数或权重。此结果支持进行一组受控训练干预，不证明方法已改善分割。

## 实测结论

| 项目 | 实测 |
|---|---|
| 完整恢复 | student/teacher、DINO/iBOT状态、AdamW及3 rank RNG恢复；optimizer step30000→30020 |
| 配对一致性 | 3 rank×20步，共60组；global/local crop、iBOT mask/indices/weights、采样链、模型前后RNG、LR/WD完全一致 |
| 历史复现 | C0全部60条loss和采样链与历史30k→100k首20步相等，loss最大差0 |
| Barrier激活比例 | 中位60.37%；范围40.53%–68.99% |
| 原始Barrier loss | 中位0.013822；范围0.005758–0.025918 |
| Barrier局部backbone梯度范数 | 中位0.235601，全部rank/step非零 |
| 相对主损失backbone梯度范数 | 比值中位6.94%；范围1.29%–20.93%，均为DDP平均前的局部范数 |
| 权重 | smokeλ=1；建议正式对照仍锁定λ=1。加权loss约主loss的0.095%（比值中位数）；没有系数搜索 |
| 数值/冻结 | 无NaN/Inf；affinity与hinge/reduction为FP32；anchor始终无grad且最终逐张量等于原EMA |
| 坐标验证 | 3 rank真实训练图像：官方forward与独立layer23接口逐token最大差0；256 patch，排除1 CLS+4 register |

当前量级下，λ=1既产生实际梯度，也未主导局部backbone梯度范数；这不是方法有效性或长期稳定性的结论。所有数字来自同一次固定参数预检，未读取MV-dev或MB1标签。

## 恢复依据与实际训练协议

完整checkpoint为`codex/ssl_diversity_d12_20261017/runs/D2/formal/checkpoints/step000030000/full.pth`，大小5,607,208,485字节，SHA256 `cce40d90b9e4f03eae9231f7516cd1a81d49e8b369db9380dbb3b30f7d82967f`。它独立于正式EMA导出（SHA `7db12e7ff8bee45166aaeeda1d94365e696ddaecf0c57a0d1611a891cf0d3eda`）；二者的teacher backbone已逐张量确认一致。完整文件有382个student状态键、382个teacher状态键、357项optimizer状态、58个参数组，以及Python/NumPy/Torch/CUDA四类RNG×3 rank。原始路径映射与完整检查记录保留私有，公开资产以相对路径和SHA定位。

没有独立scheduler或AMP scaler字段，原因已由原训练源码确认：LR是按global step计算的函数，WD恒为0.04并沿用组倍率；历史使用BF16 autocast，没有GradScaler。online teacher与model_ema是同一个对象。恢复时实际加载完整student、teacher、DINO/iBOT状态与optimizer，并恢复对应rank RNG；由global step重建函数式日程。不能把缺少原本就不存在的对象字段写成恢复失败。历史首20步逐值复现进一步验证了这一路径。

实际使用原D2三rank NCCL梯度平均实现、global batch48（每rank16）、seed0、AdamW、原始DINO+iBOT+KoLeo配置及原crop增强。KoLeo保留官方cfg权重0.1及原global-crop scaling，没有另写一个基础loss。训练输入沿用原50:50 MitoVerse/CEM池、排除清单与采样器，纯图像读取。global256、local112×8、ImageNet RGB mean/std均来自真实合并配置。上游默认train.dataset_path字段不控制这套自定义loader；实际入口明确读取D2_manifest.json。

30k续训首步LR为`2.1949885969097836e-05`，继承原100k分段续训日程，保留历史30k处的LR跳变；没有新建15k cosine或重置optimizer。完整配置见[resolved_SSL_config.json](../evidence/Pilot-C/resolved_SSL_config.json)，恢复摘要见[checkpoint_restore_summary.json](../evidence/Pilot-C/checkpoint_restore_summary.json)。本轮停在30020，不存在45k结果。

## 唯一新增损失

每张global crop取实际SSL student的第23层post-norm patch_pre_head，形状[2,16,256,1024]；冻结30k EMA在同一批global图像上取对应patch tokens。先L2归一化，以FP32计算256×256余弦矩阵。空间Chebyshev距离>2的有序patch对构成候选集，各视野以anchor相似度中位数筛选bottom-50%（中位数相等者包含）；不含CLS/register或近邻。

`L_Barrier = sum_selected max(0, A_student - stopgrad(A_anchor) - 0.05)^2 / selected_count`

mask和参考相似度均stop-gradient；全部batch按被选patch对总数归一化。新损失只加一次到官方总loss，实际更新仍只有一次backward。没有额外可学习参数，也没有官方全矩阵Gram loss、TopoCL、LAST、FFT或第二项约束。被排除或hinge未激活的affinity具有严格零直接梯度；共享token仍可能通过其他被选对获得梯度，不能将其解释为整个前景token完全不受约束。

6项GPU测试覆盖非激活loss/梯度严格零、排除对直接梯度零、非零梯度到backbone/anchor无grad、相同特征和网格、BF16 autocast内FP32计算，以及空集合。真实接口额外比较了官方forward与独立第23层输出，差异为零；诊断图与真正student tokens也在全部60个rank-step精确相等。

## 吞吐、显存与预算

以下按每步最慢rank计时，固定丢弃前5步，报告余下15步中位数。显存为PyTorch每卡peak allocated，不等于nvidia-smi含缓存和其他进程的总量。

| 配置 | ms/step | 每卡peak allocated |
|---|---:|---:|
| C0原版SSL | 767.0 | 9.53 GiB |
| C1含完整诊断，实际测量 | 1262.7 | 11.86 GiB |
| C1去除额外诊断的生产估计 | 861.1 | 未单独测量，以上述峰值为上界参考 |

冻结anchor前向中位57.6ms；剔除诊断后的C1总step较C0估计增加12.3%。测量backbone-only梯度需要单独、RNG相同的student前向/VJP：原选择性activation checkpoint不允许同一个图反向两次。诊断图不参与更新，tokens与实际forward精确一致；其额外前向、VJP及范数统计时间单独计量。861ms是扣除这部分成本的估计，不是另跑了一次关闭诊断的20步。正式阶段若保留全部诊断，必须采用1263ms的实测成本。

| 待批准工作量 | 外推墙钟时间 |
|---|---:|
| C0额外15k SSL | 3.20 h |
| C1额外15k SSL，关闭完整梯度诊断 | 3.59 h |
| 两组SSL合计 | 6.78 h |
| 再加两组Primus40k训练 | 合计8.22 h |
| 再含两套完整MV3/MB1/Linear流程 | 合计约9.23 h |
| 若全程保留完整梯度诊断 | 全流程约10.90 h |

下游时间来自Pilot B实际43.21分钟head和73.42分钟完整流程，不是本轮新训练。20步稳态观测极值外推的完整流程为9.1–10.6小时；正式排程可预留10–11小时。这里没有把短程吞吐写成已测得的15k耗时；新checkpoint写盘、启动与长程I/O波动仍会改变实际时长。

## 解释边界与停止

真实student包含iBOT mask和DropPath0.3，冻结anchor使用同一global图像的unmasked eval路径，沿用reference-teacher语义。因此60.4%的Barrier激活只说明这项损失在真实SSL条件下参与优化，不能拆分为长期漂移、mask差异或随机深度的因果贡献。非近邻低相关对没有线粒体GT语义，不能称为拓扑不变量、线粒体边界保护或已阻止远端背景坍塌。

未来15k配对最多直接回答45k时同预算C1相对C0的效果；若C0在45k尚未发生肌肉退化，不能声称阻止了100k的负迁移。冒烟模型未保存，正式阶段若获批准，两组均须重新从原始30000完整状态起跑，不继承临时30020状态。Pilot B一个有效中点不证明Linear Mode Connectivity，也不证明其HeLa回退由新细胞器结构被破坏造成。空间共模方向仍暂停，未被直接干预证伪。

工程记录：首次因遗漏原PYTHONPATH在导入阶段零更新退出；补回历史依赖目录，无安装/升级。随后C0完成20步，C1在首个optimizer update前因诊断VJP与选择性checkpoint冲突退出。修复仅涉及测量图；C0文件哈希保持原样，不重复更新；C1最终完成原定20步。失败原件留独立私有归档，运行源码C0=`abf21bb703e01393a89d238e394e8d297d8dcc9a`，C1=`2879291f0733b025c8757e292f7aa57c6f59fe5c`，各源码SHA见runtime_source.json。

本轮6项GPU单测、60组配对检查、所有数值/坐标/冻结检查均通过。历史完整checkpoint和EMA哈希前后不变；服务inactive、MainPID0、退出0，网站HTTP200。只执行C-1，未启动15k SSL、Primus、C0_Midpoint、其他融合或loss变体。后续需要用户明确批准，本轮完成即停止。

逐rank/step完整表、逐step汇总、历史对照和预算见experiments/Pilot-C；恢复、配对与停止回执见evidence/Pilot-C。公开交付只含必要代码、科学表格与脱敏证据，不包含原始图像、权重、缓存、GPU唯一标识或运行进程信息。

独立[draft PR#16](https://github.com/Polaris1010/EM-DINOv3-Research/pull/16)已创建并推送，未合并；初始结果commit `0c99770a091be0a85b9ab2ec185feb6c43056468`，PR远程head已核验。

CPU registry/仓库检查及51项unittest通过（3项既有平台skip）；6项Barrier测试在服务器GPU通过。补记交付不触发任何实验。
