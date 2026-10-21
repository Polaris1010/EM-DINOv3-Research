# D2-Coverage：复用30k状态的全覆盖采样设计

2026-10-20。状态：设计及CPU模拟完成，训练未授权、未启动。推荐C：**未访问图像全覆盖＋最小集合分布偏移的补偿重复采样**。

在D2的同一30k完整checkpoint上，继续原70k optimizer updates即可，不必重做100k。模拟在第69236步达到全部1425433张保留CEM图像至少访问一次。SSL计算成本参照历史同阶段约14.41小时、三张GPU合计43.22 GPU小时；新40k冻结Primus约51分钟，评测另计。历史D2/100k可作为有效的“同起点、同预算、不同采样方案”对照，但**不能将其称为集合分布严格不变的纯覆盖率消融**：实现全覆盖至少需要15.7352%的后70k CEM集合总变差距离，本方案达到此下界。

## 1. 读取的真实资产与边界

已读取AGENTS、PROJECT_STATE、START_HERE/交接、57条实验注册表、D12和D12-100k真实配置/训练器/loader/sampler、两轮报告、muscle review及PR8/PR9。GitHub核实PR8和PR9均为draft未合并；PR9输入head为670ed5e684f614bc945d1c92c9948bf5f865c450。最近已合并PR是初始化PR1，不能称D12结果已合并。

主要输入：

- `experiments/registry.yaml`中`D12-100k-D2-SSL`的initial_checkpoint及SHA。
- `configs/pretrain/D12_base.yaml`、`D12_100k_base.yaml`及两份protocol。
- `src/d12/sampling.py`，`src/d12_100k/{train_ssl,ssl_full_data,legacy_ssl_data,policy}.py`。
- `experiments/D12/{mitoverse_pool,cem_pool,D2_manifest}.json`及两个预算的实际sampling totals。
- `docs/D12_final_report.md`、`D12_100k_final_report.md`、`D12_Glycolytic_muscle_review.md`。

数据池不变：MitoVerse 153体积/12来源/34143合法Z，CEM1425433图/627非空collection。正式留出与已知关联排除清单不变，不读取CEM-MitoLab标签。collection不是独立acquisition的保证，未知关联仍UNVERIFIED。muscle review的假阳性现象仅作为固定诊断观察，不参与本设计任何集合选择、权重或随机种子选择。

## 2. 固定预算、符号与已完成前缀

全局batch48、3 rank×16。global index定义为j=48×step+16×rank+within_rank，step从0开始。原seed0的每索引局部Random(j)决定数据域和原始位置。

复用前30000步：1440000次全局抽样，其中CEM719071次，唯一485179图；未访问U=940254图。

沿用原D2后续数据域序列：剩余3360000个全局位置中，CEM M=1679696次、MV1680304次。故CEM占后段49.99095%，累计100k为2398767/4800000=49.97431%。不是每batch强制24:24，以避免改变历史数据域和MV序列。

对collection i定义：

- n_i：池内图像数；p_i=sqrt(n_i)/Σsqrt(n_k)。
- v_i：30k已见不同图像数；u_i=n_i−v_i，Σu_i=940254。
- a_i：历史D2后70k实际抽到该collection的次数，Σa_i=M。

所有策略使用相同30k已见集合，未修改此前图像曝光、增强或训练状态。

## 3. A/B/C算法与真实索引模拟

### A：历史有放回基线

每CEM位置以p_i选择collection，再均匀有放回抽该集合图像。实际A即历史D2/100k，不重训。给定后段次数a_i，条件期望唯一覆盖为：

E[V_i]=v_i+u_i×[1−(1−1/n_i)^a_i]。

本轮不是仅用期望估计：逐一重放实际480万global indices，来源计数与历史30k/100k实际消耗记录逐项相同，得到历史唯一图像数955756。

### B：Collection-aware Without-Replacement

**保持A每个CEM位置的collection选择完全相同**。每集合先将30k未访问ID作独立确定性洗牌队列；取完未访问ID后，对整个集合洗牌开始下一轮，每轮内部无放回。

不重置前30k已访问状态，否则会浪费分支覆盖机会。给定相同a_i，最终唯一图像数可精确写为：

V_B=Σ[v_i+min(u_i,a_i)]。

这既是模拟结果，也是固定collection配额下可达到的覆盖上限。18个大集合满足a_i<u_i，缺口合计264304，所以B不能全覆盖。

实现使用每集合/轮次的独立PCG64种子；队列离线编译成global-index→ID的不可变表。rank/worker不各自维护一个会冲突的队列；具体分发由后文统一核心完成。

### C：全覆盖＋补偿重复配额（推荐）

**阶段一：清空30k未访问集合。** 将全部940254个未访问ID进行一次确定性全局洗牌，填入后段最早的940254个CEM位置。每集合第一次覆盖配额严格为u_i，禁止重复占用这些位置。全局打散避免按collection整块连续读取；覆盖阶段最大集合份额为takemura2013的6.5152%，这是满足其剩余图像覆盖的明确配额，不是无上限放大。

此阶段结束时全池覆盖，模拟对应optimizer step69236。MV位置穿插保持原样。第一覆盖阶段的collection占比u_i/U不同于p_i，无法同时严格保持原sqrt分布。

**阶段二：对剩余R=M−U=739442次CEM抽样作配额补偿。** 为尽量回到A后70k总collection分布，令：

d_i=max(a_i−u_i,0)，D=Σd_i，r_i≈R×d_i/D。

使用整数最大余数分配：先floor，再按余数降序补足，平局按固定collection索引；保证Σr_i=R且0≤r_i≤d_i。最终后段配额q_i=u_i+r_i。

按r_i对各集合均匀有放回抽图，之后确定性全局洗牌穿插这些重复样本。重复阶段的权重是补偿后的r_i/R，**不是继续声称等于原p_i**。这一阶段不加入任何GT信息或新超参数搜索。

**最小分布偏移证明。** 欠配额集合必须增加至少S=Σmax(u_i−a_i,0)=264304次，其余集合必须让出相同次数。因而任意全覆盖方案的后段集合分布满足：

TV(q/M,a/M)=Σ|q_i−a_i|/(2M) ≥ S/M = 0.1573522828。

本方案欠配额集合q_i=u_i，其余q_i≤a_i，恰好达到下界。这不是说全部抽样属性都“最接近A”：证明仅针对**后段集合计数的TV距离**，不保证训练过程的时间分布、梯度或样本重复模式相近。

直接“全覆盖后恢复原sqrt抽样”的初版模拟TV为25.4387%；其记录保留在`initial_uncompensated_C_reference.json`。补偿配额将其降至理论最低15.7352%。这是无标签索引层面的数学修正，没有用开发成绩挑方案。

**不可忽略的代价：** 大集合仍需更多曝光，后段最大集合份额从A的1.3615%升到C的3.6470%；如takemura2013从22869次升为61259次。已经在30k全见过的小集合在第一阶段没有新增曝光，欠配额大集合在第二阶段不再获得重复配额。故存在明确的阶段性分布变化；不把它伪装成严格source-balanced不变或仅减少重复。

### 模拟结果

| 策略 | 100k唯一CEM图 | 覆盖率 | 未访问图 | 图像全覆盖的collection | 后段collection TV相对A |
|---|---:|---:|---:|---:|---:|
| A历史 | 955756 | 67.0502% | 469677 | 300/627 | 0 |
| B队列 | 1161129 | 81.4580% | 264304 | 609/627 | 0 |
| C覆盖＋补偿 | 1425433 | 100% | 0 | 627/627 | 15.7352% |

三者均访问全部627个collection；“访问过collection”和“该collection全部图像已见”不同。

| optimizer step | A覆盖率 | B覆盖率 | C覆盖率 |
|---|---:|---:|---:|
| 30k | 34.04% | 34.04% | 34.04% |
| 50k | 47.32% | 57.64% | 67.65% |
| 70k | 56.88% | 70.41% | 100% |
| 100k | 67.05% | 81.46% | 100% |

C相对理论sqrt概率的后段TV为15.7343%，A自身有限随机抽样与sqrt的TV约0.7252%。详细627集合配额、覆盖率、份额偏移见`collection_metrics.csv`；每5000步覆盖曲线见`coverage_curve.csv`；含0次访问的完整图像重复直方图见`repeat_histogram.csv`。三者单图累计访问次数最大值分别143、143、116；C不是每图只看一次。

![CPU模拟图](E:/mira_website_demo/artifacts/d2_coverage_20261020/coverage_design.png)

## 4. 30k完整恢复可行性与对照解释

注册表指定并现场核实：

`/opt/data/3_117/tanghr/codex/ssl_diversity_d12_20261017/runs/D2/formal/checkpoints/step000030000/full.pth`

SHA256：`cce40d90b9e4f03eae9231f7516cd1a81d49e8b369db9380dbb3b30f7d82967f`。

现场只读CPU mmap检查：5607208485字节；step30000；student和teacher各382个state条目；AdamW357个parameter state、58个参数组，所有已保存optimizer step均30000；三个rank的Python/NumPy/Torch/CUDA RNG均存在；DINO/iBOT center保存。CUDA未初始化，使用已有`dinov3_shared/bin/python`、Torch2.10.0+cu128。完整证据见`checkpoint_index_inspection.json`。

**没有序列化scheduler对象不是恢复缺失。** 现有训练代码本来就用step计算LR。新分支须原样继承`src/d12_100k/policy.py`：

LR(t)=1e−6+(2.598076211353316e−5−1e−6)×[1+cos(π(t−5000)/95000)]/2，t≥30000。

30000处恢复到约2.195e−5，沿用历史30k→100k的共同LR跳变，不重新warmup、不改成从0开始的100k。参数组lr_multiplier、AdamW状态、EMA0.999、loss centers等同恢复。

历史100k的initialization记录确认使用了该完全相同的30k文件，旧resume代码已严格load student/EMA/loss/optimizer、恢复各rank RNG。因此可以构成匹配分支对照，无需重训A或30k前缀。新的采样计划显式进入run identity，允许的变化仅为CEM抽样策略及其加载实现，不绕过父checkpoint的严格校验。

已访问集合由前1440000个global indices精确重放恢复；只标记**已完成optimizer updates实际消费的索引**，不把DataLoader预取但未消费图像算进已访问。历史source totals和旧采样函数逐项校验通过，前缀唯一图485179精确复现。

比较可以回答“在相同30k状态后，用全覆盖＋最小配额补偿替代有放回采样，系统表现如何变化”；不能单独区分coverage、必要collection配额偏移和时间排序各自的因果作用。若只要求严格collection分布不变，应选B，并接受81.46%覆盖上限；当前明确要求100%覆盖，因此推荐C。

## 5. 分布式确定性：模拟和训练共用一个核心

本轮新增`src/d2_coverage/core.py`，只含CPU采样核心，未接入任何训练入口。

- 历史域/位置生成由`HistoricalSampler.choose(global_index)`执行，与未修改的原sampler做500个完整record检查，并全量核对实际来源计数。
- `compile_plan`编译A/B/C的336万个后段全局位置；MV位置用−1，其余为CEM图像ID。
- `CoveragePlan.choose(global_index)`是模拟及未来loader必须调用的**同一个入口**，保留MV全部坐标、原数据域和原view_seed，仅替换CEM source/volume_id/image_id。
- 不把NumPy新采样器的RNG注入Torch/增强RNG。旧collate使用同batch首index种子，故mask随机种子/策略保持。图像内容变化后增强的实际几何结果不要求一致，但算法、输入适配及每索引seed不改。
- plan为只读mmap，含SHA256、池hash、seed、起止step，rank/worker查询顺序不改变内容。固定world_size3和batch16，沿用原索引映射，不自行按rank重复洗牌。
- sampler恢复字段仅需plan hash、consumed_step、next_global_index和world/batch（本次JSON190字节）；预取没有cursor副作用，恢复时忽略未消费请求。不得根据“worker已经取过”推进覆盖状态。
- 计划文件持久化后用哈希锁定；本地生成环境NumPy1.26.4和服务器运行环境可能不同，未来服务器直接读取相同plan，不在不同版本下静默重新洗牌。若重建必须哈希完全一致。

CPU验证已经实际完成：3 rank×2 worker进程重排请求、重开对象、checkpoint序列化/恢复、未消费预取、损坏恢复状态拒绝；每策略480个完整位置record一致（包括全覆盖阶段切换前后），所有336万ID分rank再组装一致。另6项小池测试覆盖B不能全覆盖、C预算不足拒绝、最小TV整数配额、可重复编译等。**没有执行GPU恢复训练测试，不把ID确定性测试描述为完整网络bitwise复现。**

## 6. 最少实现改动（未来获批后）

不修改任何`src/d12*`历史源码、D1/D2权重或历史报告。本轮只交付设计、核心原型、模拟及CPU测试。

未来在独立输出目录，复用现有100k执行器的副本，最少行为改动：

1. 新loader副本的`raw()`将原`choose(...)`替换为`CoveragePlan.choose(absolute_index)`；ZIP/SQLite读取、CRC、PIL RGB、RRC及MV路径完全复用。
2. 新trainer副本增加plan hash/version到identity与checkpoint，显式验证原30k父identity；保存/验证consumed_step和plan状态。继承完整恢复/优化器/LR/EMA/loss，不新增scheduler。
3. 新run配置指定独立root、C plan、同一30k路径/SHA、原100k配置，100k固定EMA选模。原controller/readout输入准备复用，不能重复此前缺manifest/cache映射的部署遗漏。

下游只为新C encoder训练一套全新同seed20261014 Primus_Multiscale，MV-Generalist-v2 156train/38dev、冻结encoder、40k、TTA OFF；固定100k做评测，50k若保存仅留档，不按MB1选checkpoint。原A head/预测复用。

## 7. 计算、吞吐、I/O和磁盘成本

**模型计算预算：** 只新增70k SSL updates，global48，计算图、crop256/112×8不变；物理GPU1/2/3，禁止GPU0。历史D2后70k为51860.8秒=14.4058小时（1.3498 step/s，约64.79基础图像/秒），参考43.22 GPU小时。计划可预留约15–17小时SSL墙钟，应以未来真实吞吐为准；不是已经测得C训练时间。

40k新head历史约50.64分钟。完整开发及MB1评测另计；此轮未进行新的GPU计时，不把其耗时伪造为已测。可为下游训练与评分另预留1–3小时，遇实际瓶颈如实报告，不改变预算/协议。

**采样开销：** 本机全480万历史位置重放约49.39秒；C计划编译约0.118秒，写出约0.028秒。全部336万ID查询约2.09秒；6000次完整C choose加baseline比对约0.128秒（包含两个sampler调用，不当作纯C耗时）。图像ID访问不是主要模型计算成本，正式image loader吞吐仍未实测。

**计划/状态：** 单方案int32 plan13440128字节（约12.82MiB），checkpoint附加JSON190字节。可选30k seen bitset为ceil(1425433/8)=178180字节；已有固定plan时不必每次重放前缀。每worker owner表约5.70MB加小型metadata；plan可共享mmap页。A/B/C三份plan与历史ID缓存共约59.52MB，不入Git。

**图像I/O估计：** 实际SQLite只读元数据核实，无图像读取。按每collection平均压缩大小×配额估计后70k CEM读取A/B约61.353GB，C约62.929GB，约+2.57%；保留池全部不同压缩成员约54.824GB。此为期望字节量，不是实测磁盘流量；更多唯一成员可能降低cache命中，随机seek和解码成本仍需未来监测。MV访问序列完全相同。无需重新下载或解压一份全库。

**磁盘：** 当前full checkpoint约5.607GB，单student/teacher backbone约1.213GB。继承每1000步保存、保留50k milestone full和最近两份full的规则，最终SSL约21.67GB，原子写入/清理前峰值约27.28GB；至少预留30GB给SSL。head、日志、特征及预测另计，MB1每模型uint8预测约2.00GB；总工作空间建议预留约40GB及原有共享cache。禁止因新分支清理历史checkpoint。

## 8. 验收、评价和结束条件

未来训练前最小验收：checkpoint SHA和step30000一致，三个rank RNG可恢复；第一batch的global index、MV坐标、view_seed、mask策略与历史一致；CEM ID符合锁定plan。只在获批后做必要短恢复smoke，保存/恢复比较后再执行固定正式预算，不增加独立研究阶段。

正式结束验收：

- 累计100k且本分支只执行70k；CEM总次数2398767、后段1679696、MV位置与A相同。
- 真实消费ID覆盖1425433/1425433，627个collection覆盖100%；实际集合次数与plan精确一致，不用预取ID冒充消费。
- full状态、plan hash、完整checkpoint与吞吐/I/O记录齐全，100k固定encoder和同seed40k head；不自动单组延长。
- 原12来源开发Dice与NSD@16nm、每来源增减/最差来源；原38区域与7组MB1全量固定诊断，包括肌肉FP、Precision/Recall、预测前景比例。MB1已暴露，不能用其GT选collection、配额或采样seed。

**结束该方向的规则：** 完成一次C续训及固定下游后就停止。如果全覆盖仅提升ID指标，而开发Dice/NSD没有共同改善，或仍呈明显来源取舍、MB1最差域与背景误检无改善，则不继续堆更多“覆盖轮次”、调collection权重或延长步数。记录覆盖不是充分解法，结束本覆盖优先方案。内部提升但跨域退化则报告局部收益，不宣称泛化成功；单decoder seed不宣称统计显著。

目前推荐：保留A作已完成对照，**只推荐C进入后续待批准实施**；B保留为已模拟的严格collection配额对照，不自动追加B训练。任何后续训练均等待用户明确批准。

## 输出与复现

代码：`src/d2_coverage/{core,simulate,test_determinism,summarize}.py`及`tests/test_d2_coverage.py`。

证据：`evidence/D2-Coverage/`中的checkpoint/index现场核实、PR引用、模拟summary、覆盖曲线CSV、627集合CSV、重复分布、确定性测试、成本估计及图片索引。C最终plan SHA256：`43d4da230fbda3c0bcd48f28bac0db18600314bfbe5ea9473361f85a7ed96316`。本地plan/图保留`E:/mira_website_demo/artifacts/d2_coverage_20261020`，大数组不入Git。

CPU复现（无图像、无训练）：在相同索引和代码下运行simulate.py，再运行test_determinism.py、summarize.py，参数分别指定repo、artifact和evidence目录。运行前确认只写新目录，原资产只读。模拟已完成，本轮无需重复执行。
