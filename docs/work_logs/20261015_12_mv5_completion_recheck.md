# MV5完成状态与GitHub交付复核

用户要求核查40k训练、已完成开发结果和报告/指标/Markdown提交推送。只读检查现有产物，没有启动训练、推理、阈值/尺度搜索或新实验。

服务器实际核验时间：2026-10-15T04:53:35.262475+00:00。40k真实更新完成，10k/20k/40k模型及resume元数据齐全，40k权重SHA256复算与记录一致。38区全部原生Z推理及全部38区/12来源物理评测完成；服务inactive、SubState dead、MainPID0、ExecMainStatus0，网站200。服务器报告、304条指标、来源表和协议SHA均与本地逐字节一致。GPU0没有被本轮使用。首次自动终止收取UTC20261015T035253Z即香港11:52:53，当前复核没有恢复执行器。

来源等权Dice：P0 0.8137414863，B100 0.8264489417，未适配MitoNet 0.6102085552，MitoNet-MV 0.7288328326。适配增益+0.1186242775，B100与适配后差距+0.0976161091。适配后6来源提升、6来源下降；Guay/Müller/Han恢复明显，OpenOrganelle/Haberl/Turner等下降。B100在12/12来源数值更高，但这是一个MitoNet适配seed对既有三个decoder seeds，不能证明独立采集泛化、MitoNet能力上限或SSL创新收益。

完整报告docs/MV5_final_report.md、逐区域/来源指标、配对、曲线、checkpoint哈希和本轮启动Markdown在结果commit09335403f53cdce18c661e35db07a484e375026e中；git ls-remote及GitHub API均确认该提交已在远端研究分支。旧266条指标、40条历史registry均未改变。早期run_index仍保留collector running及只有10k的saved_milestones，这次改为真实终止和完整三保存点，保留旧snapshot。

自动收取器在git push成功之后的PR draft/head联合断言处退出；不是训练失败，也不是结果推送失败。当前PR6 open/draft、head与结果commit一致，先前断言的瞬时原因仅凭trace无法确定，不将推测写为已证实根因。PR正文尚为启动说明，因此本次使用已授权的Polaris1010私有仓库认证更新最终说明，同时为PR head回执加入最多6次/10秒有界等待；不改变用户控制的draft状态，不重复发布或启动任何实验。新增两项假API测试覆盖短时旧head与有限等待，全部27项CPU测试及仓库检查通过后再提交。

本次复核证据：evidence/MV-5/completion_recheck.json、live_terminal_verification时间戳快照及publication_recheck.json。交付后的实际commit/PR head以GitHub回执与远端分支为准，PR不自动合并。完成后保持停止。

实际恢复完成：2026-10-15T04:56:09.412967+00:00，GitHub API逐文件确认最终报告、304条指标、原轮Markdown及训练代码均在远端结果提交；PR6正文已更新为完成结果，状态open/draft/未合并。registry --check、repository及27项CPU测试均PASS；本轮复核记录和收取器回执修正随下一提交推送，不重跑实验。
