# MV3：多来源冻结分割与 Han 整幅 GT 协议

MV2的head只在4体积/3来源接受监督，不能说明充分多来源监督的分割能力。本PR建立156 train/38 dev区域、12来源的匹配G-P0/G-B100原版Primus训练，冻结DINOv3 ViT-L/16，[5,11,17,23]，每run40k steps，3个共同decoder seeds，10k/20k/40k保存，TTA=OFF。

当前锁定MV-Generalist-v2：用户确认Han GT为线粒体真值并明确要求“取消 mask 限制，整幅 GT 参与 loss”。只对Han跳过mask，整幅GT参与采样、loss及开发指标；其他来源、train/dev/test边界和全部训练配置保持不变。GT身份确认与整幅标注完整性假设分别记录，不以模型Dice决定mask极性。

此前v1首seed两组各完成40k及完整开发推理，69/76开发指标完成时按用户更改停止。其checkpoint、结果和6条registry快照保留；v2双方从相同初始化重训，不把两种监督协议混成排行榜。旧MV1/MV2的26条历史记录、源码和结果保持不变。

验证：10项CPU测试，registry/provenance与train/dev/test边界检查；实际GPU smoke、缓存有效体素、部署SHA和训练状态见experiments/MV-3/v2及带时间戳evidence。计划预算不代表已完成训练，最终收益等待真实全来源指标。GPU0禁用；网站、环境、旧研究结果保留。PR保持draft，未自动合并。

完成六组后被动收取MV3_final_report.md、逐区域/来源指标、配对变化、失败案例、训练曲线和18个checkpoint哈希；执行器停止，不自动运行新SSL或MitoNet。
