# MB1七组官方benchmark执行记录

用户授权全部七组、八个固定run，只推理/评分。使用research-decision-discipline技能，以既有模型解决B100收益能否延续到作者benchmark的问题；不会增加训练或新诊断阶段。

官方EMPIAR元数据/212个文件清单及torch-em get_benchmark_paths已保存；论文Table1六volume3D IoU/TEM2D IoU作为独立参考。初次文件查找误遍历Zarr过久，已仅停止本轮自身find并改成有界候选路径定位；找到服务器已有EMPIAR-10982全部212原始TIFF，不使用Zarr导出替代。备用本地下载进程已按精确PID179256/命令行停止，不触及未知进程，下载partial不当作完整数据。

10项既有checkpoint/encoder SHA均实际复算通过。第一轮资产准备因错误地只接受unsigned GT而停在Lucchi++ signed实例TIFF；已接受整数实例ID，评分仍逐平面要求非负，不改变GT>0规则。失败日志保留，第二轮准备PASS。四系统原生XY/四种大小输入smoke通过，不使用GT指标挑配置。TEM100图未找到独立物理标定，nm指标留空。五Conrad来源B100池有同形状且3平面像素一致证据；Lucchi++source关联精确区域未确认，TEM未发现名称重叠，不把任何一项写成严格无暴露证明。

锁定协议和manifest见configs/evaluation/MB1_protocol.json、experiments/MB-1/MB1_benchmark_manifest.json；应生成20760条逐平面记录、56条逐run逐数据集记录及28条模型seed均值。正式推理/评分进度和实际commit将在取得后追加。MV3/4/5实验文件原样保留，网站200、GPU0禁用。最终收取本轮指标、源码、配置、哈希及所有Markdown，提交推送后停止。

正式执行代码与协议已提交推送d5adc03；服务mb1-official-benchmark-20261015.service于UTC2026-10-15 07:28:26启动，MainPID116291。原始数据106条目/2595平面，八run全量执行。初始状态已保存，O/M已完成106条目推理，官方MitoNet评分已完成；P0/B0在salivary、P1在第三GPU，1/2/3利用率87/94/91%，GPU0仍原未知进程且未使用。SSH控制连接曾重置，未造成重复任务；恢复后仅启动一次正式service，控制连接已设keepalive。继续接续同一任务，不因中途成绩调参。最终汇总与图代码后续提交，其数值仅来自固定预测。
