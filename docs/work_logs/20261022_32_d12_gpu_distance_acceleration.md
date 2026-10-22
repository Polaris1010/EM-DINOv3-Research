# 多seed剩余距离评分GPU加速

用户指出一夜未结束并要求GPU提速。此前约1小时只估计剩余head训练，没有覆盖全部推理/MB1/距离评分，时间说明不完整。8个新增head与全部预测/MB1已经完成；08:15香港剩40个距离任务，三个CPU worker正在处理约4亿体素的OpenOrganelle区域。现有GPU1/2/3各空闲约13GiB，同时运行Coverage-NR。

原选择为固定CPU exact EDT，属于调度安排过于保守。新增运行期开关，让后续既有worker使用验证过的GPU EDT；大Wei区域约16.78亿体素，float64距离场13.42GB超共享预算，因此用主存保存完整距离场，按完整空间轴把全局separable EDT送GPU，复用原CUDA内核。每条线覆盖完整轴，轴间传播在全局距离场上完成，没有空间裁剪、距离截断或近似。MONAI HD95/NSD、物理spacing、GT、预测、阈值均不变。

CPU测试56项、3项既有POSIX skip通过；完整轴批次分区和显存预算通过。部署时每个GPU worker须先通过随机/长轴/各向异性全距离场与SciPy、MONAI指标的数值等价检查。当前只完成代码，尚未声称GPU已运行。原三个在途CPU任务先完成，后续任务直接使用GPU，避免重复计算。

为了兼容NR排斥新增GPU进程的监督器，GPU期间仅再次移交原Python监督器健康检查，torchrun/ranks继续，现有多seed控制器维持心跳。GPU worker仍由原多seed控制器创建和管理，无新增独立GPU执行器。最后清理GPU进程后恢复NR监督器。旧监督移交receipt另存留档。

## 实际GPU执行

2026-10-22 08:25香港：已按用户要求将剩余HD95/NSD切到GPU1/2/3，实测286/304条完成；新服务d12-three-seeds-gpu-score-20261022.service active，只接续缺失距离评分。保留277条既有指标SHA不变，无训练/推理重跑。Han CPU实测173–182秒，GPU21–22秒（同体积不同decoder seed，非严格同预测速度基准）。全距离场数值对照误差0，MONAI HD95/NSD一致。Coverage-NR继续60440、网站200，GPU0未用。最终完整报告待评分结束，禁止重启旧服务。

原先计划让三个在途CPU任务结束后再用GPU，随后为立即提速改为受控替换评分调度器：暂停本地终态收取器、保存已完成277条的SHA、停止旧多seed评分service（没有停止NR）、确认旧GPU/CPU子进程退出、启动只计算缺失项的新service，稳定后恢复同一收取器PID121548。训练、开发预测和MB1均不重跑。旧service的正常运维停止状态另存evidence，不视为科研失败；新STATUS正在运行。

GPU实际PID和receipt见本轮快照；验证包含2D/3D、各向异性与4096长轴，完整距离场与SciPy最大绝对误差均0，MONAI fixture的HD95/NSD完全一致。每区域记录实际GPU、后端和源码SHA。当前大Wei的host-backed生产执行尚待其进入队列，不能提前声称其速度。

真实阶段结束时间（香港）：所有head=03:39:06，全开发预测=06:14:52，全部MB1=07:32:28。此前1小时仅对应03点之前的剩余head，没有给出完整端到端ETA；本轮明确修正说明。CPU距离评分是过于保守的调度选择，已实际切到GPU。
