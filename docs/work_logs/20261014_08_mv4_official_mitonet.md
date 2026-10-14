# MV-4官方MitoNet同图像对照

用户明确收尾MV3，授权下一项唯一实验MV4。复用MV3所有三个seeds的G-P0/G-B100预测和指标；仅官方MitoNet_v1对同38区域/12来源做推理。没有重新训练、head延长、TTA更改或SSL模块。

22:13:45香港时间只读核实MV3 complete、service inactive/MainPID0、网站200；GPU1/2/3可用，GPU0既有外部PID未碰。已有MitoNet目录只有dataset，没有权重/empanada环境。

外网连接出现可复现问题：服务器DNS解析失败；本地Git克隆和整文件下载被中断，MitoNet完整221241200 bytes权重只收到约8MB，因此未使用该截断文件。改为本地有界分段续传，完成后强制核验Zenodo发布者MD5，再上传至独立MV4目录。下载不是科学诊断，也不改变协议。源码与官方配置小文件已收到并锁定SHA。

代码已接入官方原Engine2d/Preprocessor定义及官方empanada-dl0.1.7发布包。采用官方二维batch工作流、统一tile1280、阈值0.5、scale1、默认center/NMS、TTA OFF；原生全Z，推理不读取GT或annotation mask值。只在评分时读取现有GT/有效区；Han整幅GT，其他来源不变。逐层panoptic和instance输出保留，不声称3D tracking。

独立server_root为`/opt/data/3_117/tanghr/codex/mv4_official_mitonet_20261014`；新venv只在该目录安装额外依赖，不更改基础MV3环境或网站。环境初次核心导入发现dask缺失，按官方依赖补入该隔离目录。实际通过、smoke、推理、评分状态另追加证据；当前不得声称MitoNet结果已完成。

完成后输出MV4_final_report.md、完整指标/差距/失败图和资产哈希，提交推送代码/Markdown/小表。停止，不自动启动下项研究。当前分支codex/mv4-official-mitonet，从已完成MV3 commit578a30e派生；PR2仍是MV3的draft，未合并。
