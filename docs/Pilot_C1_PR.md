完整D2/30k恢复与唯一Selective One-Sided Gram Barrier预检完成。C0/C1各20次更新、6项GPU单测通过；60组输入/RNG/日程相等，C0首20步loss与历史逐值相同。

固定m0.05、bottom50%、Chebyshev>2、layer23、λ1。Barrier激活中位60.4%，局部backbone梯度比中位6.94%。C0 767ms/step，C1去除完整诊断的估计861ms（含诊断实测1263ms）；完整15k配对+下游估约9.23h。吞吐外推不是正式训练结果。

包含唯一损失、GPU测试、真实SSL适配、有界执行器、独立配对复算、完整表格、报告和工作记录。失败工程尝试均未增加相关分支optimizer更新；已完成C0保持原SHA，未重复训练。原始历史62条registry不变。

Student原iBOT/DropPath与eval anchor不同，激活不能归因为长期退化；低相关对不具有GT语义。未来45k同预算比较不自动证明阻止100k遗忘。只完成C-1，服务停止；不启动15k SSL或Primus。独立draft PR基于Pilot B结果分支，不自动合并。
