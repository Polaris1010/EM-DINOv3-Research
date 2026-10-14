# MV-4B定向输出路径复核

用户明确要求仅MV4B：四个已有区域先比较官方Semantic Only与既有Panoptic；输出差异明显后才扩展全38。保留全部MV4历史结果，不重训/重推DINOv3，不GT调阈值，不扩大灰度或尺度搜索。修正上一轮立即转向robust SSL的决策顺序，先排除输出路径混杂。

23:44前服务器只读检查：MV3/MV4 complete、MV4 service inactive/MainPID0，网站200；GPU0原未知PID2470228、网站PID1971241未动，physical1/2/3可用，磁盘充足。实际版本0.1.7源码确认：没有合格中心时instance_cells全零；Panoptic merge只保留实例分配成功的thing前景。官方Semantic Only通过thing_list=[]绕过thing前景分配，同时保留stuff_area64语义总面积规则。这是直接源码事实，尚不能推断实际4区域的差距来源。

新目录和分支独立。原官方forward概率在同一次Semantic Only运行中捕获，不自写sigmoid/后处理；raw>=0.5与official mask分别保存并逐平面断言官方总面积过滤后相等。所有GT访问仅在独立score中，旧Panoptic混淆矩阵复算须与MV4原记录相同。实际试验状态与测量另追加。
