# GPU优先执行偏好与Coverage-NR评分入口

用户要求后续全部使用GPU加速。已将GPU优先要求写入AGENTS、START_HERE、NEXT_ACTION与新窗口交接：训练、推理、HD95/NSD的可加速核心计算使用物理GPU1/2/3；GPU0禁用。大距离场采用主存保存和完整空间轴CUDA分块，不默认退回纯CPU。CPU保留I/O、调度、表面/ignore准备、统计汇总和必要数值验证。

## 实际代码

检查发现Coverage-NR benchmark入口仍有MV2_GPU_EDT=0。将它改为必须在GPU评分上下文内运行。开发评分同样接入独立gpu_scoring模块，复用上一轮D12三seed验证过的float64全轴EDT；GPU显存充足时走原GPU实现，显存不足时按连续块传输、CUDA转置和完整轴变换，工作区256MiB。没有局部截断、近似距离或指标精度变化。

有ignore mask时沿用原表面提取与safe-mask语义，只将两次EDT交给GPU。每次评分输出backend、物理GPU、源码SHA和full-device/host-backed调用计数；GPU不可用即明确报错，不静默走CPU。

修改仅限src/coverage_nr/{gpu_scoring,downstream,benchmark}.py；train_ssl.py、run.py、采样、增强、loss、预算、checkpoint和既有指标均不变。原入口备份保存在本轮服务器evidence/GPU_scoring_source_before_20261022。部署仅替换尚未调用的评分子进程入口；训练器保持原样。原计划不重启controller，实际校验误触保护后的恢复情况如下。没有新实验。

## 验证与交付

小型合成数据逐项核对二维/三维、各向异性、长轴4096、主存完整轴分块的距离场，以及有/无ignore的HD95和NSD。CPU只用于一次性小fixture参考值；真实后续评分使用GPU。等价性结果和部署源码SHA见evidence/Coverage-NR/GPU_scoring_equivalence_20261022.json与GPU_scoring_deployment_20261022.json。

本轮代码、GPU偏好、证据和Markdown进入codex/coverage-nr-p0-training、draft PR11，不自动合并。下一窗口延续现有训练和既定评测，不为这次运维改动重训或改协议。

## 实际中断与恢复

独立GPU合成校验在物理GPU3执行，触发旧controller对未知GPU PID的保护，停止了其自身SSL ranks；错误为Unrelated new GPU process。该操作错误已向用户明确说明。原任务最终可见94450步，最后完整checkpoint94000，SHA256 033f04058bf85c728eac4e450b87965544dd97bcb666b1a0628bb7f1070a3f07；450步需要重放。旧失败STATUS与progress已保存GPU_test_interruption_20261022.json。

数值校验已通过：4个完整距离场最大误差均0，有/无ignore的HD95和NSD一致。没有重复这组GPU验证。

新增resume_existing.py，仅复用原controller并将正式SSL启动改为严格94000 full checkpoint恢复，原train_ssl.py完全不改。加载原student/EMA、optimizer、DINO/iBOT centers、三rank RNG、采样计划状态；身份和checkpoint SHA由原训练器严格检查，LR仍按绝对step沿用连续100k日程。无新初始化、无预算延长、没有重训练已有D1/D2。部署收据UTC07:39:57，新接续服务coverage-nr-p0-gpu-resume-20261022.service。后续验证只读服务/文件，不再并行启动控制器不认识的GPU进程。

未来GPU测试必须作为controller管理的子进程，或在无训练占用时运行；不得独立启动后再期待保护器自动接纳。

UTC07:41:55已确认initialization=full_state_resume、starting_step94000，严格身份校验通过，GPU1/2/3均100%，网站200。此时旧progress仍为中断前94450，故run_index暂写恢复点94000，不将旧值冒充恢复后的新进度。恢复后elapsed_seconds只代表接续段，整轮耗时需结合中断前84238.7秒及接续时间报告，避免误报为仅6000步耗时。

UTC07:42:48已确认新progress94080、loss有限、GPU1/2/3均100%，网站200，原任务恢复生效。实际验证：GPU完整距离场4项误差0，有/无ignore指标均一致；CPU仓库validator、registry check及51项unittest通过（3项已有POSIX skip）；git diff --check通过。
