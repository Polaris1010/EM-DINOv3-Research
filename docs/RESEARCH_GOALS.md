# 研究目标

基于DINOv3 ViT-L/16开发多来源EM通用2D线粒体分割模型，研究能够提升可迁移表征的自监督机制，并在公平监督、数据暴露、推理和计算协议下与官方MitoNet比较，形成可核验会议级研究成果。

当前科学问题仅为：**B100k相对于P0的表征收益，能否在更广泛的MitoVerse开发来源中成立？** B150k为已存在的续训参考。固定原版Primus readout和九个MV-1 head，先收取已在运行的无TTA开发扩展结果。

阶段目标不是组合更多模块。Gram、HRA、LAST、TopoMask或其他SSL机制均需后续明确实验授权与可检验假设。失败诊断保留负面证据，不能改写成功。

未来严格inductive泛化与官方MitoNet公平对照，需要独立处理图像隔离、acquisition重叠和锁定test协议；它们限制最终论文结论，不阻塞当前标明transductive的开发任务。
