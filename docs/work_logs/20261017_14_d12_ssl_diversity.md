# D1/D2多来源SSL数据扩展与MitoNet原始留出依据

用户授权2026-10-17第一轮D1/D2：同自然DINOv3初始化、30k、无新loss，固定首seed Primus40k/156train/38dev；先核实MitoNet是否在SSL使用test，并继承其留出原则。已应用research-decision-discipline技能。

原文Methods明确从CEM1.5M/CEM-MitoLab排除benchmark与关联图像，Lucchi++整个train也排除；不是“SSL不能用训练图像”。本轮共享测试/开发图像及已知关联资产留出。旧B100有无标签评测图像暴露，保持历史，不叫监督标签泄露。D1与B100因预算/暴露边界差异仅作描述性参考；D2−D1才是匹配的数据扩展比较。

服务器已有CEM全包61833797020字节、1592753图。中央目录651集合目录/640非空，未把集合当独立acquisition。SQLite随机访问索引和逐成员CRC，排除13个具名留出关联集合后保留1425433图/627集合；未知哈希别名交叉关系UNVERIFIED，不开展全库审计。MitoVerse新池153volume/12source/34143合法Z，源清单不修改。CEM监督标签不读取。

真实loader smoke通过，两组global[32,3,256,256]/local[128,3,112,112]、原生图像读取/抽样检查通过。35项CPU测试通过。训练smoke第一轮因误用Torch2.5分割环境缺官方SSL接口而导入失败，未训练、无环境升级；已切换回B100既有dinov3_shared Torch2.10环境重试，结果待核实。所有失败和重试文件保留在本轮新根。

实现统一source/acquisition/volume/Z sampler、CEM sqrt(n)集合权重、固定30k调度与10k/30k导出；原官方增强和loss不改。新encoder必须新训匹配Primus，原MV3代码函数只在新输出根调用，cache只读复用。控制器顺序运行两组三GPU SSL，再完成两组首seed训练/开发/固定MB1诊断，之后报告并停止。不会自动100k、D3、更多seed或新诊断。

本轮分支codex/d12-ssl-diversity，从已推送MB1提交2388de9分叉；初始状态/源码提交、正式执行及GitHub交付证据将在取得后追加。原49条registry与历史证据保持，新增本轮4条SSL/head记录。

第二轮训练smoke两组均完成：2真实optimizer steps、三GPU global batch48、rank参数最大差0，student与EMA非零更新，完整checkpoint已保存；无GT训练、无参数搜索。loader和训练smoke分别验证数据接口与真实分布式优化，没有把smoke权重当作30k或正式下游encoder。

## 正式启动与交付

正式源码4029f2681a0ebf5e511aa37eba14462712ce10e4已提交推送；服务器executed_sources.json逐文件SHA与本地代码/配置一致。正式service d12-ssl-diversity-20261017运行中，MainPID3370458；初始快照D1 20/30000、global batch48，GPU1/2/3利用率93/92/100%，GPU0未用，网站200。D2及后续head/评测按有界控制器接续，未声称正式训练完成。

私有仓库draft PR8已创建，base为codex/mb1-official-benchmark，不合并。35项CPU测试、D12留出/概率/预算验证、53条registry及历史49条保持检查通过。独立有界收取器scripts/watch_d12_results.py仅等待本轮既定任务，终止后收取实际结果、更新本记录/交接、检查并提交推送；不启动新实验。认证仅在进程内存，不落盘。若桌面/网络中断，服务器任务仍独立执行，下一窗口从服务器STATUS接续收取，禁止重复启动。

结果收取器已现场确认连通，UTC2026-10-17T02:48:35.539465，状态WAITING_FOR_AUTHORIZED_PIPELINE；其不启动实验，只等待既定任务终止后收取/检查/提交。启动后复核D1已到180/30000，三GPU持续工作；30k及下游结果尚未完成，不能提前给出研究结论。

## 最终自动收取

D1/D2终止状态 failed，UTC 20261017T032244Z。 保留实际产物，查看STATUS与服务器阶段日志；未完成部分不得声称完成，不自动改参/重启。

本收取器只读取已有结果，不启动实验或修改服务器数据。全部本轮小型文本产物、代码和Markdown一并提交推送，draft PR不合并。
