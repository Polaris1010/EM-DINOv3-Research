# D1/D2：第一轮多来源 SSL 数据扩展

用户2026-10-17明确授权D1/D2匹配30k与首seed冻结Primus40k。原B100/MV1至MV5/MB1保持，禁止GPU0、网站修改、依赖升级、新loss或自动100k/D3。当前事实以experiments/D12/run_index.json与服务器STATUS为准，不把smoke或计划写成正式训练完成。

本轮唯一匹配问题是D2−D1：同初始化、同预算、同MitoVerse留出池下，固定50%CEM是否改善分割。D1−历史B100同时改变100k→30k预算及测试图像暴露边界，不能单独证明采样效应。没有擅自添加第三组训练。

## MitoNet原始依据与采用的边界

[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9883049/)的Benchmark dataset characterization及STAR Methods的Benchmark curation明确：六体积/100TEM及相关图像都从CEM1.5M和CEM-MitoLab排除，包括整个Lucchi++训练集、OpenOrganelle其他果蝇神经数据及同C.elegans项目。训练图像可用于无标签预训练，不等同测试图像参与训练。来源证据见evidence/D12/mitonet_ssl_holdout_evidence.json。

D1/D2共同移除既有正式test、不明混合test资产、MV3开发区域及已知MB1关联资产；Casser/Xiao等仅保留不交叉Z，不能称独立acquisition。CEM按可识别名称排除Guay、Müller、MitoEM、Kasthuri及cos7等留出关联集合。未对未知哈希集合实施全库图像去重，它们与MV3开发/测试的精确关系仍UNVERIFIED。作者原始CEM的MB1留出依据采用其明确声明，不伪称重新认证全部像素。原MV3监督数据不改变，因此完整系统也不能自动升级成作者相同的采集体隔离协议。

## 实际数据与采样

完整CEM包已存在：/opt/data/3_117/tanghr/MitoNet/dataset/cem1.5m.zip，61833797020字节，1592753图，651集合目录（640非空）。无重新下载，无CEM-MitoLab标签访问。移除13个具名关联集合后1425433图/627非空集合；完整清单、理由、ZIP与SQLite索引SHA见experiments/D12/cem_pool.json。以ZIP central-directory偏移实现无损随机访问，所选成员解压后核验CRC32，不把内部CRC当成出版方SHA。

MitoVerse原212volume清单不修改，新池153体积/12来源/34143合法Z，见mitoverse_pool.json。D1仅该池；D2以0.5概率选MitoVerse、0.5选CEM。MitoVerse内部source等权→已声明acquisition或volume代理等权→volume等权→合法Z与XY裁剪；CEM集合概率sqrt(n)，集合内部图像等概率，gamma0.5。目录数或slice数不等同独立采集体。

MitoVerse保留既有512图像crop与reflect padding；CEM保留原生PIL RGB（通常224），不预先插值或补成512，同一官方RandomResizedCrop输出global256/local112。所需格式适配和native FOV差异随数据扩展披露，不能解释成纯数据数量效应。

## 实际执行约束

SSL复用B100实际环境dinov3_shared（历史Torch2.10.0+cu128）、官方DINOv3 commit91ffb6803c74aeb6927209b82560142743763dd7、同自然ViT-L/16权重SHA8aa4cbdd…，没有B100续训。沿用DINO/iBOT/0.1KoLeo、global batch48（三GPU各16）、seed0、AdamW和5000warmup；30k cosine horizon，两组完全相同。10k/30k导出，固定30k做下游，10k不用于选模。模型/优化器/EMA均新建于本轮根，不覆盖旧checkpoint。

无标签loader smoke已通过，35项CPU测试通过。首次训练smoke误用分割环境Torch2.5，因官方SSL接口缺失在导入阶段失败；日志保留，不升级该环境。第二轮改回B100实际既有环境，D1/D2均真实完成2步、global batch48、三rank参数最大差0、student/EMA均有非零更新，checkpoint保存通过。smoke不是正式30k结果，正式状态见run_index与服务器STATUS。

正式两组顺序使用GPU1/2/3，保证原global batch；完成后并行重新训练同seed20261014的两个Primus40k，使用既有156train/38dev缓存和原函数，验证encoder冻结、head初始化及采样hash匹配，完成全部76条开发指标。随后固定评测MB1七组作为已暴露开发诊断，不据此更改配比、checkpoint或超参数。原Panoptic基线和MV3旧预测不重跑。

服务器新根：/opt/data/3_117/tanghr/codex/ssl_diversity_d12_20261017。控制器有独占锁、72h上限、heartbeat、网站与未知GPU进程保护，仅清理自身子进程；失败保留产物，不自动更改参数。结束生成D12_final_report.md、逐来源/区域/配对/benchmark指标、checkpoint SHA和历史输入前后快照。实际吞吐/显存与抽样分布保存在rank日志及sampling_totals。

正式结果、代码和本轮Markdown提交推送后停止；PR保持draft，不自动合并，不启动新loss、100k或更多seed。
