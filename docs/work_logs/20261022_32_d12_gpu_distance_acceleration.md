# 多seed剩余距离评分GPU加速

用户指出一夜未结束并要求GPU提速。此前约1小时只估计剩余head训练，没有覆盖全部推理/MB1/距离评分，时间说明不完整。8个新增head与全部预测/MB1已经完成；08:15香港剩40个距离任务，三个CPU worker正在处理约4亿体素的OpenOrganelle区域。现有GPU1/2/3各空闲约13GiB，同时运行Coverage-NR。

原选择为固定CPU exact EDT，属于调度安排过于保守。新增运行期开关，让后续既有worker使用验证过的GPU EDT；大Wei区域约16.78亿体素，float64距离场13.42GB超共享预算，因此用主存保存完整距离场，按完整空间轴把全局separable EDT送GPU，复用原CUDA内核。每条线覆盖完整轴，轴间传播在全局距离场上完成，没有空间裁剪、距离截断或近似。MONAI HD95/NSD、物理spacing、GT、预测、阈值均不变。

CPU测试56项、3项既有POSIX skip通过；完整轴批次分区和显存预算通过。部署时每个GPU worker须先通过随机/长轴/各向异性全距离场与SciPy、MONAI指标的数值等价检查。当前只完成代码，尚未声称GPU已运行。原三个在途CPU任务先完成，后续任务直接使用GPU，避免重复计算。

为了兼容NR排斥新增GPU进程的监督器，GPU期间仅再次移交原Python监督器健康检查，torchrun/ranks继续，现有多seed控制器维持心跳。GPU worker仍由原多seed控制器创建和管理，无新增独立GPU执行器。最后清理GPU进程后恢复NR监督器。旧监督移交receipt另存留档。
