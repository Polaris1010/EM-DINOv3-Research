# 2026-10-14：MV-3多来源监督Generalist训练

用户明确要求执行MV-3，仅P0/B100k、frozen encoder、original Primus、多来源监督、共同40k预算和3 decoder seeds；停止旧9个head扩展验证，不追加SSL模块、Linear Probe或新全面审计。

已执行：只读核实MV2完成1692行/188区域/12来源，GPU1/2/3没有研究占用，网站HTTP200；GPU0既有未知进程未触碰。复用212-store metadata、既有split/QC和Han有效区修复。建立并锁定MV-Generalist-v1（156 train、38 dev、12 train/dev来源）；准备194个新区域cache指针，复用旧数组只读，生成Han正确有效mask与缺失Wei-R train cache。所有区域有效体素存在，未静默删减队列。

实现：新loader、frozen Primus训练器、现有evaluator适配、有界6-run执行器和report脚本。共同固定40k选模，保存10k/20k/40k；首seed完整dev后再补两seed。先真实GPU smoke，再正式训练，新输出全部写MV3目录。具体源码/配置及路径见MV3_PROTOCOL。

验证：新增split交叉、test混入、OpenOrganelle分组与Han非零编号mask回归检查；共9项CPU测试通过。GPU smoke及正式进度必须以随后实际回执为准；当前尚不声称模型完成或已有新Dice。后续会将实际snapshot、运行状态和最终报告追加到本记录及registry。

Git：工作分支 `codex/mv3-generalist-training`；本轮代码、配置和Markdown一并commit/push，实际commit/PR写入后续交付记录。保留原26条历史实验记录和MV1/MV2原文件。每轮状态不以聊天代替。
