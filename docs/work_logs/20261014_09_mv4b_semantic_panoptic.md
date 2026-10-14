# MV-4B定向输出路径复核

用户明确要求仅MV4B：四个已有区域先比较官方Semantic Only与既有Panoptic；输出差异明显后才扩展全38。保留全部MV4历史结果，不重训/重推DINOv3，不GT调阈值，不扩大灰度或尺度搜索。修正上一轮立即转向robust SSL的决策顺序，先排除输出路径混杂。

本轮初始服务器只读检查：MV3/MV4 complete、MV4 service inactive/MainPID0，网站200；GPU0原未知PID2470228、网站PID1971241未动，physical1/2/3可用，磁盘充足。实际版本0.1.7源码确认：没有合格中心时instance_cells全零；Panoptic merge只保留实例分配成功的thing前景。官方Semantic Only通过thing_list=[]绕过thing前景分配，同时保留stuff_area64语义总面积规则。这是直接源码事实，尚不能推断实际4区域的差距来源。

新目录和分支独立。原官方forward概率在同一次Semantic Only运行中捕获，不自写sigmoid/后处理；raw>=0.5与official mask分别保存并逐平面断言官方总面积过滤后相等。所有GT访问仅在独立score中，旧Panoptic混淆矩阵复算须与MV4原记录相同。实际试验状态与测量另追加。

## 四区域实际测量与停止扩展决策

23:53:39核验pilot complete，service inactive/MainPID0/Resultsuccess，4/4原生全Z推理及4/4路径比较，网站200、GPU0未用，MV3/MV4保护回执前后相同。Guay Semantic/Panoptic Dice都0.014252511409；Müller high_c1都约2.34698e-7（不是Müller两个区域的来源均值0.000168）；Casser都0.940288544329，Wei都0.935873512097。四区域官方Semantic前景丢失为0；raw>=0.5与旧Panoptic统计一致。Müller仅official Semantic stuff_area64过滤329个GT背景像素，TP仍2，Dice差4.53e-12。

单次extension_decision=false：明显路径差异未出现，不扩展38。实例分组造成Guay/high_c1极低分这一解释在当前参数/区域下不受测量支持；灰度/尺度/真实域外识别仍是未操纵候选。不启动后续审计/训练。

准备阶段registry重复追加被CPU检查发现，执行前修正为40条；首次本地tar创建被默认沙箱限制，首次launch没有包，未启动worker。实际pilot运行代码8a8b757。初始53f430f与修复8a8b757已推送，未把准备失败写成科学成功。

报告收尾跨到2026-10-15香港时间。四个完整区域raw probability>=0.5与旧Panoptic逐像素XOR0已在服务器独立复核，不仅是统计相同；概率float32与全Z形状/文件资产已保留。四张八面板图已查看Guay/Müller，标签重叠仅以CPU重新排版，不改数值或推理。历史完整精度敏感性分析：去Guay/Müller后Δ+0.099292991，再去Han后+0.061319361；两来源贡献61.735104%，不删除正式来源。

最终CPU核验PASS：4完整区域、32行系统对照（24既有DINOv3+4既有Panoptic+4新Semantic）、混淆矩阵/PR/Dice/前景算术、概率/阈值与历史Panoptic关联；原39条registry逐项未变、历史MV4报告逐字节未变。registry/repository检查与21项测试PASS。新报告和本轮Markdown/交接提交推送，后续只做发布回执，不启动新实验。

实际交付：完整结果commit222b1f64292c46764713836822c93fdb38c74487已推送；GitHub API确认private、Polaris1010、PR4 open/draft、head与本地一致，base为未合并MV4分支codex/mv4-official-mitonet。PR链接https://github.com/Polaris1010/EM-DINOv3-Research/pull/4；没有自动合并。最后收尾提交固化此发布回执。
