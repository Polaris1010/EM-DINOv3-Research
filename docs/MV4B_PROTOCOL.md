# MV-4B：Official Semantic vs Panoptic

唯一问题：Guay/Müller极低的MitoNet前景覆盖是否主要来自实例分组，而非原语义概率？复用官方完整v1权重、原uint8缓存、官方norms0.57571/0.12765、confidence0.5、scale1、padding16、tile0、TTA OFF。第一批固定Guay vol1、Müller high_c1、Casser既有dev、Wei既有dev的全部原生Z。已有Panoptic输出只读复用，不重跑DINOv3，不读取GT调参。

使用原官方Engine2d.update_params(semantic_only=True)，保留官方后处理。官方RenderEngine.infer在同一次模型forward中提供sem=logits_to_prob(sem_logits)；只增加只读概率捕获，保存float32概率及>=0.5的raw mask，同时保留官方Semantic Only mask。后者会将类1视为stuff，per-plane总面积小于64时过滤；这不等于每连通组件64像素过滤，不能在报告中混淆。

先比较4区域Dice/Precision/Recall/前景比例、Semantic Only前景在Panoptic中消失的有效像素/比例、中心数为零的平面。正式扩展依据输出路径差异，先保存一次decision再运行，不能依据GT选择阈值或scale。若差异明显，扩展同38区域，用原MV3物理距离evaluator补HD95nm/NSD@16nm；保留Semantic与Panoptic两项成绩及原DINOv3三seed结果。若差异很小，按4区域结果结束，不扩大审计。

服务器独立MV4B目录，仅复用既有MV4venv推理、原meddinov3评分，不安装包。GPU0禁用，只用physical1/2/3；单独服务/锁，每阶段最长2小时，网站/未知进程保持，历史输入hash/stat前后核验。大型概率/二值预测留服务器，小表/代码/报告/本轮Markdown与交接提交推送，PR保持draft不merge。

此轮同时纠正研究表述：MV4是“MitoVerse监督适配后的DINOv3 Generalist vs未经本项目标签适配的released MitoNet逐层2D Panoptic前景”，不是公平独立测试上的预训练方法全面优越。可归因于现有匹配encoder对照的开发增益仍是B100−P0约+0.0127。不覆盖MV4历史报告，以本轮独立补充报告说明输出路径和敏感性分析。

来源证据：src/mv4b/upstream/来自实际0.1.7发布wheel；Engine2d来自MV4 hash锁定官方文件。用户复核原文保留在docs/reviews/MV4_user_review_20261014.md。执行/指标状态以新run_index和带时间戳回执为准。
