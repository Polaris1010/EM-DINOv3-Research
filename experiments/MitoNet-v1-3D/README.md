# 官方 MitoNet-v1 三轴推理

运行位置为 .98 研究服务器的 /opt/data/3_117/tanghr/codex/mitonet_v1_orthoplane_20261120。运行 launch.sh 时，物理 GPU 1 和 3 各处理一个分片。模型来自已注册的 Official-MitoNet-v1 发布权重，SHA256 为 487d63dd1deb971a3e374785b350673ee5427e0ffcd44880bf9a54f6c2c4bf04。原权重与三维源文件的哈希均在启动时核对。

从锁定的 empanada-napari 源码载入 Engine3d、tracker_consensus 与配套归一化。推理依次进行 XY、XZ、YZ 三向预测和切片匹配，随后执行官方实例共识。在原始网格上将最终实例 ID 大于零的体素视为语义前景。所有体积采用相同的引擎默认参数，scale 1、median kernel 5、confidence 0.3、每轴过滤 500 体素与跨度 4；最终共识 pixel vote 2、cluster IoU 0.75、最小体积 200。无来源专用参数。三维分支使用单机同步生成器代替 Qt 的线程调度，运算函数体沿用官方实现。

MV3 的 38 个开发区按各自完整长方体推理，再按原 valid_z 和 mask 评分。MB-1 的 6 个体数据使用同一三轴设置，TEM-100 的图像使用历史官方二维入口独立推理。二维参照来自对应固定 checkpoint 的历史重测。各组语义指标包括 Dice、IoU、Precision、Recall。MV3 区域先汇总有效体素，再按来源聚合。MB-1 保留逐切片均值与六个体的体素池化 IoU。二者可在聚合上产生数值差异。

每个完成区域或图像写入 scores，每个三维体另在 pred 保存打包的二值预测和原始形状。完成后运行 report.py，将有效真值数量逐项与原二维数据核对，生成 comparison.csv、volumetric_metrics.csv 和 report.md。此对照使用发布权重及统一默认推理参数，独立于本项目另行训练权重的论文复现实验。
