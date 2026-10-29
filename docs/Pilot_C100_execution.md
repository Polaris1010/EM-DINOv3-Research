> 已完成并停止，最终结果见[Pilot_C100_final_report.md](Pilot_C100_final_report.md)。以下为按时间保留的执行记录。

# Pilot C100：45k完整状态至100k配对续训

用户已批准执行。固定两份现有45k full各续训55,000步，原30k EMA anchor、lambda1、margin0.05、bottom50%、Chebyshev>2、Layer23、历史100k LR/WD函数、配对数据/增强/RNG保持。两新100k EMA各自匹配原版冻结Primus40k（seed20261014）、Linear23原500步及全部38dev/七组MB-1，完成报告和提交推送后停止。

## 准备时状态（审批超时已解决）

PLANNED / 已授权但未运行。训练0步，新服务器目录/服务尚未由本轮创建。未建立已认证SSH连接，因此本轮没有重新核实GPU占用、网站或远端checkpoint；输入SHA来自已完成C-2登记，执行前仍须真实核验。源码14文件已编译，61项CPU测试通过（3项既有Windows skip），新增6项覆盖错配parent、陈旧optimizer、anchor误改及RNG边界。

自动审批对SSH凭据安全输入连续两次返回“审查未在截止前完成”；Git暂存提权也连续两次同样超时。这些是审批超时，不是认定动作不安全。认证重试/安全终端选择已通过异步问题交用户确认；不重复索要密码。尚未提交或推送本轮代码，不声称训练已开始。

## 已完成的实现

src/pilot_c100独立于历史C-2。Barrier、production adapter、Primus/Linear/benchmark/GPU scorer与C-2逐字节相同。新代码只改变45k父状态选择、100k固定终点、36h有界调度及七系统汇总；C1始终从登记的原30k EMA加载anchor，绝不改锚到45k。

完整恢复验证student/teacher、DINO/iBOT、optimizer step45000及三rank RNG；续训第一步的RNG/采样链必须与对应C-2最后一步连续。全部55k更新逐rank保存输入/RNG/日程hash，最终核对165,000组rank-step，验证每次Barrier同一backward参与且没有诊断VJP。

正式下游只评价100k EMA及固定40k head；Linear AP不作门槛，不利用MB-1 GT选择参数。输出七系统完整逐区域/逐来源/逐图指标，绝对100k C1−C0为主，45k→100k变化差为描述，并比较Pilot B中点。Recall与Turner/HeLa/OpenOrganelle损益一并报告。

## 待继续的既定步骤

认证恢复后只读核查两45k full及原30k anchor的实际SHA、设备1/2/3空闲和磁盘容量，然后封存源码commit/协议，部署独立目录并启动有界服务。旧C-2服务不重启；GPU0禁用，网站和未知进程受保护。按实测吞吐估计全流程约26–30h。没有自动延长、调参或增加Loss授权。


2026-10-28 18:14 HK 已实际启动：认证与Git审批恢复；两45k full和原30k anchor、原cfg/trainer/数据manifest实际SHA均核验，optimizer各45000，3 rank RNG/采样链齐备；网站200，三允许设备空闲、磁盘充足。源码初始3b0eb40因新增policy模块与legacy policy重名在import阶段0更新退出，原件已保存在独立attempts；改名continuation_policy并增加命名空间回归测试，62项CPU测试通过（3平台skip），服务器CPU导入及原44999 LR逐值复现通过。固定参数、预算与核心文件没有变化。正式源bd2ea834cb13163495c0d2d22c8df6467c76aa50，C0已45060，本轮60/55000，原45k RNG与采样链连续性检查通过。C1及下游排队，尚无100k新开发/MB1指标。GPU0未用、网站200。新有界服务active，不重启旧C-2服务。


2026-10-30香港完成复核：两组100k SSL、匹配40k Primus/500步Linear23及全部38dev/七组MB1均完成，165000组rank-step严格配对无重放。418份小型输出逐个SHA核验、七系统来源/组均值独立重算一致。C1相对C0：开发Dice−0.001007、MB1+0.049892，肌肉Dice+0.298020/AP+0.298814/Recall+0.061474、远端FP6.4958%→0.4171%。相对中点开发+0.002127、MB1+0.028946。100k支持保留固定Barrier候选主线；OO/Turner/C.elegans损失详见最终报告。完整流水线26.12h；服务inactive/退出0，网站200，GPU0未用，六历史输入重新SHA一致；本轮停止，不自动新训练。
