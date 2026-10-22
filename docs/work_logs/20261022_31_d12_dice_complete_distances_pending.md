# 多seed训练/预测完成，先交付完整Dice

2026-10-22 08:09香港：多seed新增8个head40k、304开发预测与全部8组MB1均完成；完整三decoder-seed Dice已输出docs/D12_three_seed_Dice_readout.md。100k：D1开发0.818676±0.000324、MB1 0.476961±0.002890；D2开发0.839914±0.001345、MB1 0.463563±0.010832，D2内部优/MB1低的方向三seed一致。剩HD95/NSD评分254/304，服务运行，非完整终态。Coverage-NR继续约59.15k、网站200，GPU0未用；不要重复启动。

## 本轮执行

仅查询现有service与结果，未训练、未推理、未改配置。服务phase=exact_CPU_distance_metrics；8个training_complete与8个inference_complete均PASS，每组38区域及2595张MB1图。原监督器已恢复，SSL继续。CPU评分从08:08的251条到08:09的254条，仍在前进。

直接复用全部overlap和旧seed指标计算三seedDice，逐seed匹配初始化/采样hash并通过完整区域计数。保存现场JSON、五张小CSV与Dice报告；本地独立重算12来源/7组等权聚合和样本SD一致。原59条登记保持；新增8个head标已完成，但整项仍VERIFIED_RUNNING。

## 判断更新

D2的内部优势及MB1总体低于D1均三seed同向；D2续训至100k两类均值均改善。D1续训MB1此前单seed下降，三seed均值实际+0.004030且方向2/3正，旧单seed解读需修正。肌肉D2三seed均值0.443746→0.196840继续下降。详情及逐benchmark见Dice报告。

## 后续

只让既有CPU距离评分、最终汇总和被动收取继续；最终完整报告尚未生成，不能称全部指标已交付。当前已完成的Dice/MB1结果和本轮Markdown提交推送现有draft PR12。不新增训练或超参搜索。
