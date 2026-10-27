# 2026-10-27 Pilot B：固定D2轨迹中点

用户批准只执行alpha0.5单Encoder；匹配Linear23不能作为取消Primus的门槛。实际500步Linear23、40k Primus、38dev及2595张MB1均完成。

固定中点恢复了肌肉的分割能力，并提高了MB1七组均值，但未完全保留100k在HeLa、OpenOrganelle等来源的收益。MV3来源等权Dice为0.838631（较100k -0.002748），MB1为0.505823（+0.030070）。保留Trajectory-Lerp05作为单Encoder轨迹融合基线；本轮停止，不搜索新系数。

肌肉Primus Dice恢复103.3%的早晚期损失，AP恢复82.4%；Recall 0.380833→0.391853。HeLa、OpenOrganelle、Guay均高于30k、低于100k。Linear与Primus在OpenOrganelle方向不一致，未引入线性硬门槛。

新EMA哈希、10k/20k/40k head、原初始化和采样序列已记录；两历史肌肉mask复算差异均0。历史61条registry不变，仅新增Pilot B。GPU0禁用，网站200，原始数据与历史权重只读。

运行源码commit2d3bcf4及完整元数据保留私有。自动审批曾拒绝中途公开运行信息，该推送未执行。最终从A2完成commit6aadf40建立干净公开分支codex/pilot-b-results-public，只提交最终脱敏结果与必要代码，私有运行提交不推送。科学CSV字节不变，元数据公开投影记录前后SHA；仅common.py路径定位改为可配置，其余七个运行文件不改。

完整结果与方法限制见Pilot_B_Trajectory_Lerp05_Report.md。有效秩变化不等于错误机制；空间共模仍暂停。下一项若获授权，可做同30k起点、同预算Vanilla与单一稳定性约束；本轮没有启动。

独立算术复算、registry、仓库检查和unittest结果随最终交付归档。独立[draft PR15](https://github.com/Polaris1010/EM-DINOv3-Research/pull/15)已推送，未合并；初始结果commit `370b6bd48023034b11b705dc49c6f2f3e506c162`，远程PR head已核对。服务inactive/退出0，停止，不运行其他实验。

提交交付前独立复算、registry和仓库校验全部PASS；51项unittest通过，3项既有Windows平台skip。最终60文件结果commit已推送；这次补记实际PR链接与交付回执，不运行任何实验。
