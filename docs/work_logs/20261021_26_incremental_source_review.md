# 仅metadata的IsoOrg / QuantEM增量审查

日期2026-10-21香港时间。读取用户附件完整要求，保留P0→100k MitoVerse+CEM设计；不下载图像、不改sampler、不训练。沿用research-decision-discipline，明确未匹配标识不等于独立性证明。

复用当前D12两池与历史212-volume catalog、最终CEM目录/UID来源表、IsoOrg组文件和历史审计，新增QuantEM官方657行CSV（revision99db73c，SHA6688bedd...），只读取候选EMPIAR/Figshare/Zenodo metadata与公开目录/Zarr属性。627保留CEM中578可精确连接来源metadata，49未匹配；已有TEM/STEM/SBF，不把模态名称当新增独立域。

输出6组合重叠表、657行QuantEM交叉表、40个IsoOrg scan名称表、当前CEM和MV成员表，以及15个公开候选入口（8优先/7条件性）。同DOI研究记录不直接传播为同acquisition；版本DOI只连接记录，不证明像素相同。旧622个MV外IsoOrg子体积中104个已有最终CEM来源证据，剩518/18名称候选，仍非独立来源证书。

核实QuantEM粒度问题：EMPIAR12174为43切片×16拼图，11958输出合并两次SBF；未将其表格acquisition列直接累加。四个IsoOrg优先候选也是QuantEM来源，只计一次。pancreas-2/3因Mueller研究家族关联暂不推荐；kidney-2属性路径404仅记入口待定位，不声称数据下架。

所有输入SHA、原始小型metadata与访问回执保存到evidence/Data-Expansion-20261021；脚本仅CPU/metadata，无网络或图像导入。历史结果、checkpoint、registry和P0设计不改。报告、日志、交接与现有PR10同步提交推送，PR不合并，结束STOP。

实际验收：metadata重放与输入SHA/15候选成员关系PASS；报告与六组合计数一致性PASS；registry一致性与repository validation PASS（115 provenance、664历史文件、57实验、2007文本文件）；51项CPU单测48通过、3项既有POSIX测试Windows跳过。约1.02MB新增metadata证据，没有图像资产。
