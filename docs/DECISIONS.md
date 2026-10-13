# 已确定的决策

| 决策 | 依据/本轮核验 | 后续约束 |
| --- | --- | --- |
| 本轮只做接管与首PR初始化 | 用户明确任务；本轮执行receipt | 不启动训练、推理或调度，不改网站/旧文件 |
| 正式readout为原版MedDINOv3 Primus_Multiscale | 原版上游/运行源码SHA及AST审计、MV-1 head metadata | 使用[5,11,17,23]，不带FFT gates/LAST |
| MV-1/MV-2均TTA=OFF | MV-1 locked protocol + MV-2 manifest与source | 不继承CMCC/CMCT四翻转TTA |
| B150k是100k完整状态的续训 | parent SHA与实际result/sidecar一致 | 额外50k，1e-6；不得写成独立从零150k |
| 保留拓扑诊断负面结论 | Stage2A FAIL与Stage2A2 GT诊断 | 不自动启动T1/TopoCrop长训 |
| 新B_full的Gram OFF/HRA NOT_RUN | config、trainer和实际结果 | 不混入旧P1的Gram尝试 |
| 当前是transductive开发资产 | 完整SSL manifest与exposure记录 | 不声称未见图像泛化；B_clean留待未来 |
| MV-2无需重新启动 | 本轮service/process/output快照 | 下一项收取既有无TTA结果 |
| 保留原始冲突记录 | CM factory registry与完成results冲突、旧Stage2A2 STATUS滞后 | 以实际运行证据优先，明示冲突 |
| 研究仓库与网站仓库分离 | 网站工作区既有修改/混合资产 | 只提交精选研究元数据和源码，不推整个网站目录 |

## 阻塞分类

BLOCKING：GitHub认证无效，阻塞remote/PR；当前样本图像与GT不能对应时仅阻塞该样本；九个既有head缺失或身份不符时阻塞对应推理。

NON_BLOCKING：full optimizer安全读取限制（当前仅评测teacher/head）；额外dinov2环境盘点超时（本任务使用已核验meddinov3）；尚未证实独立acquisition（当前transductive开发声明）。

FUTURE：最终论文的B_clean隔离、独立采集体泛化和官方MitoNet完整公平对照。未发现需要为了它们重训或暂停当前已运行MV-2的理由。
