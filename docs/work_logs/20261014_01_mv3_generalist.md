# 2026-10-14：MV-3多来源监督Generalist训练

用户明确要求执行MV-3，仅P0/B100k、frozen encoder、original Primus、多来源监督、共同40k预算和3 decoder seeds；停止旧9个head扩展验证，不追加SSL模块、Linear Probe或新全面审计。

已执行：只读核实MV2完成1692行/188区域/12来源，GPU1/2/3没有研究占用，网站HTTP200；GPU0既有未知进程未触碰。复用212-store metadata、既有split/QC和Han有效区修复。建立并锁定MV-Generalist-v1（156 train、38 dev、12 train/dev来源）；准备194个新区域cache指针，复用旧数组只读，生成Han正确有效mask与缺失Wei-R train cache。所有区域有效体素存在，未静默删减队列。

实现：新loader、frozen Primus训练器、现有evaluator适配、有界6-run执行器和report脚本。共同固定40k选模，保存10k/20k/40k；首seed完整dev后再补两seed。先真实GPU smoke，再正式训练，新输出全部写MV3目录。具体源码/配置及路径见MV3_PROTOCOL。

验证：新增split交叉、test混入、OpenOrganelle分组与Han非零编号mask回归检查；共9项CPU测试通过。GPU smoke及正式进度必须以随后实际回执为准；当前尚不声称模型完成或已有新Dice。后续会将实际snapshot、运行状态和最终报告追加到本记录及registry。

Git：工作分支 `codex/mv3-generalist-training`；本轮代码、配置和Markdown一并commit/push，实际commit/PR写入后续交付记录。保留原26条历史实验记录和MV1/MV2原文件。每轮状态不以聊天代替。

11:10:38香港时间服务器快照：两组smoke已PASS，decoder初始化SHA均为 `2ec39fb096d1a0eceb3124a9e08549d0c6ac82665dd71f8e2bc7b5a8d56f48ec`，12来源采样一致；encoder未变化，ignore loss不变且ignore梯度零，checkpoint往返通过。两组正式run均已到100/40000 steps，采样链SHA与来源计数一致；service active，phase first_seed_train_and_infer，网站HTTP200。该100step快照不能写成最终结果。

实际来源清单与有效体素已收取：194区域，193区域复用旧缓存只读、1区域新增缓存，Han有效mask新建在MV3目录。文件见MV3_actual_supervision.json及MV3_source_coverage.csv。

首次启动在训练前因GPU0宿主机PID不可见失败；修复仅限制初始化进程发现到GPU1/2/3并兼容namespace中不可见的既有PID。GPU0进程未操作，失败log保留服务器新目录。初始代码commit `27c6316`、启动器修复 `08de7d0`；[草稿PR #2](https://github.com/Polaris1010/EM-DINOv3-Research/pull/2) 已建立。27c6316的GitHub CI成功；后续提交CI实时核验，不能套用旧成功。
