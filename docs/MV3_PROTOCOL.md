# MV-3：Generalist Frozen Segmentation Training

2026-10-14用户决策：结束现有9个head的扩展验证，立即进入多来源监督训练。本轮只比较G-P0和G-B100；不新增Linear Probe、Gram、HRA或全面数据审计。旧只读初始化限制被该任务授权取代，GPU0、网站及历史保护约束继续适用。

MV-2已在服务器完成1692条核心指标，12来源等权Dice为P0 0.3492、B100k 0.4067、B150k 0.4102。当前head仅监督4体积/3来源，这些结果不能代表多来源充分监督后的generalist上限。不继续200k SSL，B150k本轮不训练。

## 新数据版本

`configs/segmentation/MV-Generalist-v1/manifest.json` 在smoke前锁定：156 train区域、38 dev区域，双方均覆盖12来源。正式test继续排除；保留现有映射的官方validation，Müller high_c1从旧MV1 train移回官方val。OpenOrganelle130 train/22 dev，按既有18个sample组的固定hash顺序留出4组。自建MV2 dev可以进入train，历史结果原件保持不变。

Casser/Xiao用同一体积不相交Z区段，记录共享acquisition限制，不要求虚构独立采集体。Turner全部microns1区域dev、全部pinky变体train，避免已知realigned版本跨两侧。Wei-R采用已有经典train0:400、val400:500。Wei-H和Lucchi的现有合并test坐标无法解释，整资产排除；不开展新坐标审计。

Han直接复用2026-09-17已核验修复：非零末端编号mask为有效区，零区设ignore=-1，包括其中正标签；不把未标注区当背景。只有有效像素参与CE/Dice和正式指标。MV2曾按相反mask约定计算，历史数字完整保存，不能将其极低Dice直接视为encoder负迁移。

实际数组与有效体素记录来自loader准备的 `MV3_actual_supervision.json`。这是本轮训练输入准备，不是重新全面数据审计。已有source registry、proofreading和QC被复用。缺失acquisition身份仍为UNVERIFIED，不阻塞当前transductive开发训练。

## 锁定对照

官方DINOv3 ViT-L/16全程冻结，官方原版Primus_Multiscale，层[5,11,17,23]，无FFT gates。固定native-grid normalization和MV1滑窗256/stride128、tile batch16、softmax平均阈值0.5，TTA=OFF、无后处理。

G-P0/G-B100共享source-aware监督采样、随机初始化head、seed和40000 optimizer steps。AdamW、batch1、lr1e-4、warmup100、共同cosine floor0.1、CE+valid soft Dice。原有固定预算选模规则延续为双方固定40k，保存10k/20k/40k；中间8个确定性有效Z切片仅监测，不分别挑best step。

先seed20261014两组：短smoke检验12来源loader、ignore loss/gradient、frozen encoder、checkpoint往返和推理；通过后从头正式训练。两组全来源38个区域正式指标完成后，补seed20261015/20261016。共6个正式run，不因低分改配置或为某encoder单独追加预算。

## 有界执行与交付

运行目录 `/opt/data/3_117/tanghr/codex/mv_generalist_v1_20261014`。loader仅复用旧cache为只读输入，新mask/cache和全部模型输出写新目录。执行器仅允许物理GPU1/2/3的UUID，监控网站HTTP200与未知GPU进程；失败时只停止自己的进程组。新数据准备和执行器均不改网站或已安装环境。

最终输出MV3_final_report.md、每区域/每来源指标、同seed P0→B100配对变化、前景比例与失败案例、18个checkpoint SHA和训练曲线。共同40k预算不等于已证明收敛；仍在改善时明确记录，不自动延长。

MV2/MV3分开保存。监督扩大收益仅通过相同新heldout区域与新有效mask上的旧已存预测进行描述性比较，不重新运行旧9个head；不把改变划分的总均值混成同榜。缺少旧heldout预测的Müller high_c1不虚构基线。当前输出不用于独立生物采集泛化结论。

完成后停止。下一项科学任务是官方MitoNet同图像/有效GT对照，后续用户任务再执行；不自动启动SSL创新。
