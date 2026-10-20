# Glycolytic muscle既有预测复核与采样覆盖

用户授权：复用D1/D2的30k、100k既有预测，输出原图/GT/预测对照，定位新增假阳性；解释均匀采样比例；提交推送结论和本轮Markdown后停止。禁止调阈值、搜尺度或训练。

实际执行：

- 独立目录`/opt/data/3_117/tanghr/codex/d12_muscle_review_20261020`执行CPU脚本，原图/GT及4个prediction文件只读，未加载模型或调用GPU。
- 302层×4模型=1208条混淆计数全部匹配既有指标，图像/GT哈希匹配；预测前后SHA256和mtime不变。
- 生成9张原图/GT/四模型对照、1张128px细节、1张空间分布；明确6个均匀切片与3个错误挑选切片。
- 图上大量FP在背景细密纹理内部，片状/方块状，不只是真值边界扩张；新增FP 6227810，93.7120%距同层GT>180nm，302/302层均有新增FP。
- 单次检查完成，未发现输入/评分错配证据，不把机制归因到唯一模块，不继续拓展诊断。
- 同seed和索引重放ID，400个原采样函数检查点通过，全部来源计数与实际记录一致。100k唯一ID覆盖D1 MV97.63%，D2 MV92.75%/CEM67.05%；保留集合均100%。未读取训练图像。

报告：`docs/D12_Glycolytic_muscle_review.md`。代码：`src/d12_review/`。证据/CSV/图像SHA索引：`evidence/D12-muscle-review/`。PNG保留服务器及本地`E:/mira_website_demo/artifacts/d12_control_20261017/muscle_review`，不将原图/大型预测放入Git。

遇到的问题：历史NPY头shape使用list，新NumPy拒读；只读解析，在内存转tuple，原文件不写入。首次本地coverage输出到artifact被拒，计算不变，重跑到仓库证据路径成功。

验证：哈希、预测不变、全切片指标和采样计数断言通过；代码CPU编译、registry一致性、仓库检查在提交前执行。未修改训练代码或历史结果，不重复训练测试。

本轮代码、结论、CSV/JSON、Markdown及交接入口提交当前`codex/d12-100k-continuation`分支、推送private仓库，沿用draft PR9不合并。实际commit以Git回执为准。完成后STOP，没有训练、阈值/尺度搜索、额外推理或后台服务。
