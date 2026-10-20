D2的100k历史运行只覆盖67.05%的保留CEM图像。本PR只设计并模拟复用同一30k完整checkpoint的采样分支，没有训练、推理或读取真实训练图像。

使用真实索引和历史数据域序列：A重放67.05%；保持逐位置collection选择的B达到81.46%；推荐C先覆盖未见ID，再补偿重复配额，69236步达到100%。C相对历史后70k集合计数的TV为15.7352%，恰好达到全覆盖所需下界；不是严格集合分布不变的覆盖率消融。

30k full checkpoint路径来自registry，现场CPU验证SHA、student/EMA/optimizer和三rank RNG。提交同一套拟用于未来loader的采样核心、CPU模拟、627集合指标、覆盖/重复分布、确定性测试、成本估计及D2_Coverage_Experiment_Design.md。原D1/D2配置、权重、报告和registry保持不变。

核心6项测试及真实计划3rank×2worker恢复测试通过；每策略480个完整位置record（含覆盖切换）及336万ID分rank再组装一致。全CPU单测51项，48通过、3项既有POSIX测试在Windows跳过；registry及仓库验证PASS。训练仍须明确批准，draft PR不合并，不启动任何新模型、loss或额外实验阶段。
