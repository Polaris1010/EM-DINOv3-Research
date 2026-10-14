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

23:07:15正式启动最终整图协议（protocol SHA `66db27f82a731c2edb81a74c681b2cd00a747abecfcc3c0c6e9f9082033ebe4b`），三种尺寸smoke全部通过。23:07:49快照：service active/MainPID3901502，推理3/38、评分0/38，physical1/2/3利用率85%/35%/43%，网站200、GPU0未用。源文件SHA均与部署commit`db2ae4e2007c85b9badee722fd331fad138d17c2`一致。启动wrapper残留旧短commit元数据已修正，原件保留deployment_metadata_before_correction.json；只改回执，运行代码/协议未动。注册状态更新VERIFIED_RUNNING。继续执行至最终报告，再停止。

## 最终收尾与研究判断

23:15:04香港时间实际complete：38/38推理和评分，228条DINOv3指标原样复用，service inactive/MainPID0/Resultsuccess，网站200，GPU0未用；MV3 protected inputs前后逐项相同。收取93份小文件及12张六面板图，不把原图、权重、预测数组入Git。CPU结果核验PASS（266行、38区域/12来源、算术及SHA、原DINOv3指标未变）。

来源等权Dice P0=0.813741、B100=0.826449、MitoNet=0.610209；B100−MitoNet=+0.216240，8胜4负；MitoNet在Casser/Mekuc/Turner/Wei更高。保留Guay/Müller的极低官方输出，不改阈值或GT极性。已人工查看Guay/Müller/Haberl/Han/Wei/Kunduri/Turner图，具体误差及Han未标注区不能当生物学假阳性的限制写入报告。自动Turner最坏平面输入全黑的图原样保留，不能据此概括来源弱边界，不开启新审计。

生成MV4_final_report.md和12图本机HTML入口，保存机器生成报告原件与图片SHA。当前状态、交接、NEXT_ACTION、PR3描述和registry更新为完成，明确历史段落不作为当前任务。下一项科学决策倾向跨来源稳健性，但未开始研究；原有限执行器已经正常终止，本轮完成后停止。

最终交付核验：21项CPU测试PASS，repository115精选+664历史来源、39条实验记录校验PASS，registry --check PASS；与MV3 commit578a30e比较，原38条注册记录逐项不变。最终报告本机/服务器SHA相同，运行源SHA与部署记录相同。代码准备、实际启动与运行记录commit1940bb2/d5b4a96/404e1b9/db2ae4e/d0c40b9已推送；最终指标/报告/Markdown随本轮完成提交推送，PR3仍draft/open，不自动merge。

最终结果commit92ee2197ab9e183386fd44c9c5a995561c119c23已实际推送并通过GitHub API核验：private=true、Polaris1010、远端branch/PR3 head与本地一致；PR3保持open/draft，base为MV3分支，未合并。发布回执见evidence/MV-4/github_final_delivery.json。收尾提交仅固化此回执和工作记录。
