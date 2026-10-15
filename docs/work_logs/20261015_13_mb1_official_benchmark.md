# MB1七组官方benchmark执行记录

用户授权全部七组、八个固定run，只推理/评分。使用research-decision-discipline技能，以既有模型解决B100收益能否延续到作者benchmark的问题；不会增加训练或新诊断阶段。

官方EMPIAR元数据/212个文件清单及torch-em get_benchmark_paths已保存；论文Table1六volume3D IoU/TEM2D IoU作为独立参考。初次文件查找误遍历Zarr过久，已仅停止本轮自身find并改成有界候选路径定位；找到服务器已有EMPIAR-10982全部212原始TIFF，不使用Zarr导出替代。备用本地下载进程已按精确PID179256/命令行停止，不触及未知进程，下载partial不当作完整数据。

10项既有checkpoint/encoder SHA均实际复算通过。第一轮资产准备因错误地只接受unsigned GT而停在Lucchi++ signed实例TIFF；已接受整数实例ID，评分仍逐平面要求非负，不改变GT>0规则。失败日志保留，第二轮准备PASS。四系统原生XY/四种大小输入smoke通过，不使用GT指标挑配置。TEM100图未找到独立物理标定，nm指标留空。五Conrad来源B100池有同形状且3平面像素一致证据；Lucchi++source关联精确区域未确认，TEM未发现名称重叠，不把任何一项写成严格无暴露证明。

锁定协议和manifest见configs/evaluation/MB1_protocol.json、experiments/MB-1/MB1_benchmark_manifest.json；应生成20760条逐平面记录、56条逐run逐数据集记录及28条模型seed均值。正式推理/评分进度和实际commit将在取得后追加。MV3/4/5实验文件原样保留，网站200、GPU0禁用。最终收取本轮指标、源码、配置、哈希及所有Markdown，提交推送后停止。

正式执行代码与协议已提交推送d5adc03；服务mb1-official-benchmark-20261015.service于UTC2026-10-15 07:28:26启动，MainPID116291。原始数据106条目/2595平面，八run全量执行。初始状态已保存，O/M已完成106条目推理，官方MitoNet评分已完成；P0/B0在salivary、P1在第三GPU，1/2/3利用率87/94/91%，GPU0仍原未知进程且未使用。SSH控制连接曾重置，未造成重复任务；恢复后仅启动一次正式service，控制连接已设keepalive。继续接续同一任务，不因中途成绩调参。最终汇总与图代码后续提交，其数值仅来自固定预测。

## 完成收取与停止

全部7benchmark/8run完成，20760条逐图指标、56条run×benchmark、28条模型均值和9张固定代表图。G-P0 Dice 0.457938；G-B100 Dice 0.502665；Official-MitoNet-v1 Dice 0.712154；MitoNet-MV Dice 0.240936；B100−P0 +0.044727。B100超过官方MitoNet：；B100超过P0：fly_brain, hela_cell, lucchi_pp, TEM-100；MitoNet-MV相对官方下降：c_elegans, fly_brain, glycolytic_muscle, hela_cell, lucchi_pp, salivary_gland, TEM-100。

服务器终止快照UTC 2026-10-15T08:01:15.067219+00:00，service inactive/MainPID0，网站200；MV3/4/5历史输入保护快照逐字节相同，checkpoint无创建/覆盖，未使用GPU0，未修改环境。完整预测、Panoptic和未舍入JSON留服务器；代码/配置/精确计数CSV/指标/报告/哈希证据在Git。代表PNG按仓库纯文本策略保留本地artifacts/mb1_control_20261015/figures及服务器，Git内有选择规则与SHA。

独立CPU复核从精确TP/FP/FN/TN重算全部20760条二维重叠指标和六体积IoU，核实8run同GT像素、七组等权及三seed平均。TEM100 TIFF均无可靠ImageJ/OME/PhysicalSize标定，ResolutionUnit=NONE、1/1分辨率不冒充nm，HD95/NSD留空。完整物理指标用原固定evaluator，6组宏平均，失败inf保留。

实际执行源码SHA见executed_sources.json：推理/评分对应d5adc03，控制器/报告对应1a08b3d；最后收取提交仅新增结果与独立核对/交接记录。原41条registry保持一致，新增8条MB1完成记录。适用检查结果将在最终交付前追加。任务完成后停止，不启动新训练、诊断、更多seed或搜索。

报告和9图首次视觉复核发现总标题与子图标题拥挤；仅在CPU上重排固定9图并澄清结论用语，代码282240c，未重跑推理/评分、未改变选图或任何指标。新report.py SHA及时间见evidence/MB-1/report_finalization.json。新版图已人工查看Fly brain、肌肉、TEM，布局可读；数值和CSV SHA保持不变。

三匹配seed的七组宏Dice提升依次为+0.055341、+0.047083、+0.031758；Fly/HeLa/Lucchi三seed同向提升，C.elegans/肌肉/唾液腺三seed同向下降，TEM方向不一致。官方MitoNet七组均高于B100，适配MitoNet七组均退化；不能支持广泛超越官方模型或严格未见图像泛化的主张。

最终检查通过：独立精确计数/覆盖/汇总/服务器输出SHA核对PASS；update_registry.py --check一致；validate_repository.py检查115项原始证据、664项历史文件、49条实验记录和1751个文本文件PASS；31项CPU单元测试PASS；git diff --check无错误。全部本轮报告、指标、代码与Markdown交接进入同一分支，draft PR7保持未合并；实际最终提交/远端SHA以Git和交付回执为准。
