# MV-4官方MitoNet同图像对照

用户明确收尾MV3，授权下一项唯一实验MV4。复用MV3所有三个seeds的G-P0/G-B100预测和指标；仅官方MitoNet_v1对同38区域/12来源做推理。没有重新训练、head延长、TTA更改或SSL模块。

22:13:45香港时间只读核实MV3 complete、service inactive/MainPID0、网站200；GPU1/2/3可用，GPU0既有外部PID未碰。已有MitoNet目录只有dataset，没有权重/empanada环境。

外网连接出现可复现问题：服务器DNS解析失败；本地Git克隆和整文件下载被中断，MitoNet完整221241200 bytes权重只收到约8MB，因此未使用该截断文件。改为本地有界分段续传，完成后强制核验Zenodo发布者MD5，再上传至独立MV4目录。下载不是科学诊断，也不改变协议。源码与官方配置小文件已收到并锁定SHA。

代码已接入官方原Engine2d/Preprocessor定义及官方empanada-dl0.1.7发布包。采用官方二维batch工作流、统一tile1280、阈值0.5、scale1、默认center/NMS、TTA OFF；原生全Z，推理不读取GT或annotation mask值。只在评分时读取现有GT/有效区；Han整幅GT，其他来源不变。逐层panoptic和instance输出保留，不声称3D tracking。

独立server_root为`/opt/data/3_117/tanghr/codex/mv4_official_mitonet_20261014`；新venv只在该目录安装额外依赖，不更改基础MV3环境或网站。环境初次核心导入发现dask缺失，按官方依赖补入该隔离目录。实际通过、smoke、推理、评分状态另追加证据；当前不得声称MitoNet结果已完成。

完成后输出MV4_final_report.md、完整指标/差距/失败图和资产哈希，提交推送代码/Markdown/小表。停止，不自动启动下项研究。当前分支codex/mv4-official-mitonet，从已完成MV3 commit578a30e派生；PR2仍是MV3的draft，未合并。

## 已交付代码与隔离环境

初始实现、配置、官方小文件来源/SHA和本轮Markdown已提交推送commit `1940bb2093769f290db363c83ab9b9f720e80276`。独立draft PR3建立并核实open，base为codex/mv3-generalist-training；不自动合并。registry生成39条，之前38条历史记录保留。21项CPU测试、完整repository及registry核验通过。

隔离环境实际核心导入PASS：Python3.11.9/Torch2.5.1，empanada-dl0.1.7，NumPy2.2.6/Numba0.61.2、cztile0.2.0及官方所需OpenCV/Dask。初始cztile2.0 API不兼容，已固定含fixed_total_area_strategy的0.2.0；基础meddinov3中的NumPy2.4.4未卸载/修改，新NumPy只写入venv。依赖版本和实际回执见environments/MV-4_requirements.txt及evidence/MV-4/environment.json。

MV2原始validation/gpu_edt在新环境实际导入成功，validation SHA `143c5403458ca3fad215880ac7afd14e4fd066fbee8342051a0cc49a083e4d54`与MV3每条指标一致。gpu_edt自行读取原有只读gpu_deps，未重新安装或重写其实现。当前仅准备完成，科学smoke/38区域执行尚未开始，等待完整权重MD5。

## 完整权重与实际推理接入

22:50后完整221241200 bytes权重已通过Zenodo发布者MD5，SHA256 `487d63dd1deb971a3e374785b350673ee5427e0ffcd44880bf9a54f6c2c4bf04`；已上传服务器独立MV4目录。初次smoke发现接入代码对ScriptModule调用requires_grad_不支持，改为逐参数冻结，不改官方源定义/权重/后处理；失败证据保存，修复commit404e1b9已推送。

小图真实推理PASS。大图的官方Tiler在NumPy2/Numba0.61编译失败；仅将新venv改为NumPy1.26.4/Numba0.59.1/llvmlite0.42.0，编译通过后又暴露官方Tiler空overlap输入的IndexError。没有任何正式预测或GT指标。停止走分块兼容路径，直接采用官方默认整图tile_size=0；23:03香港时间最大4096×4096图像在GPU3实际PASS，不读GT。旧tile协议、成功小图/失败大图和日志全部存入服务器evidence/tiling_attempt_superseded；本地旧协议存configs/evaluation/history/MV-4_tiling_attempt_20261014.json。

最终固定官方整图协议，scale/阈值/NMS/实例后处理和MV3数据不变。新controller仅用venv推理，评分/报告继续调用原meddinov3环境，避免NumPy兼容调整影响物理评测。正式38区域尚待新固定协议smoke和同一有界执行器启动；实际状态以新快照为准，未声称MitoNet Dice已得到。
