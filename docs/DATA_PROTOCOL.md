# 数据与推理协议

本轮catalog记录212 volumes、13 sources；只读列举到212个Zarr stores。SSL正式manifest为 `configs/pretrain/pretrain_all_mitoverse.json`，本轮字节SHA确为 `3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9`。历史48,722 legal Z slices、157 OpenOrganelle volumes和636代表切片审计属于既有证据，不等于全部chunks完整checksum。本轮未读取全部原始chunks。

B100k/B150k使用所有历史train/validation/test/excluded的无标签图像池，属于transductive SSL development assets。训练loader/manifest快照有无标签使用记录，不能把无标签暴露直接称为监督label leakage，也不能将已暴露图像评测声称为严格inductive泛化。

## MV-1

以 `configs/segmentation/MV-Dev-v1.json` 的精确region/spacing/array/label契约为准，4个监督train volumes和4个dev volumes，guay21/jiang25/muller21三个来源。四train与四dev ID不重叠；acquisition独立性未证实。三个decoder seeds为20261013、20261014、20261015，模型均P0/B100k/B150k，head20000 steps固定选择。

固定encoder、原版Primus、uint8灰度复制RGB后 `(x/255-0.5)/0.25`，不做physical resampling。native XY滑窗256，stride128，tile batch16，重叠mean softmax，argmax，无postprocessing，TTA=OFF。具体训练sampling/augment/loss/optimizer和评测细节见locked protocol，禁止根据CMCC/CMCT替换。

## 现有MV-2

manifest实际188个volume/ROI、12来源，19个reserved test IDs排除；复用原MV-1四个dev结果，不重训head。来源包括152个OpenOrganelle stores；数量不等于152个独立acquisitions。

manifest混合180个exploratory_custom_dev、2个published_val、2个official_ME2_val_native_store、2个MV1_custom_dev_of_published_TRAIN和2个official_classic_MitoEM_val_region。不能称188个都是官方validation。需要在后续结果中分开报告官方验证与自定义开发队列，并登记缺少native映射的官方区域、excluded项、unknown acquisition。

原MV-1 inference函数、normalization、256/128、TTA=OFF和九个head保持不变。manifest的threshold=0.5在二分类softmax下对应原argmax，记录原实现，不能自行改阈值。带valid mask的扩展区域按实际实现的valid surfaces、ignore adjacency处理；这属于新数据的计分契约，后续报告须明示，不能静默把CMCC的surface单位套过来。

每来源最终报告Dice、Precision、Recall、HD95 nm、NSD@16nm，以及volume/ROI数量、known/unknown acquisition数量、每seed统计和失败/缺失样本。NSD16nm是预设工程容差，不宣称生物校准。未知acquisition ID不能填0，也不能把ROI count称独立采集体count。

CMCC/CMCT原版历史结果使用四翻转TTA，HD95 voxel-equivalent、NSD@1；保留其原units与head选择规则，不能并入MV-1/MV-2协议。

发布者proofreading声明和数值QC不等于独立人工重标注完整性认证。CEM-MitoLab是CEM1.5M子集这一历史判断，以及MitoVerse/CEM/IsoOrg共同来源候选，仅为本地审计登记；共同来源不等于相同ROI，本轮不下载CEM1.5M。现有CSV来源索引见 `experiments/history/data_governance/`。

未来B_clean需要新的图像隔离协议和明确授权，目前PLANNED，不阻塞本轮或现有transductive开发任务。
