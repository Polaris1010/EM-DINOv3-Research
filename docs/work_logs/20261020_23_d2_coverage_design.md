# D2-Coverage设计与CPU模拟

用户只授权阅读真实项目、核实30k恢复状态、比较A/B/C及无GPU索引模拟；禁止启动训练或修改原实验。已应用research-decision-discipline，按同起点、最小变量、明确证据边界给出单一推荐。

读取AGENTS/PROJECT_STATE/registry、D12真实代码配置/报告、muscle review、PR8/9（均draft未合并，PR9输入670ed5e）；最近已合并PR1。

从`D12-100k-D2-SSL`注册项定位30k full，服务器只读CPU加载验证SHA cce40d90…，student/EMA/AdamW/loss centers与3rank RNG完整。没有scheduler对象，原训练器按step计算LR，继承原分段公式即可。没有CUDA初始化、模型前向或训练图像读取。

真实seed/index模拟：A覆盖955756（67.05%），B1161129（81.46%），C1425433（100%），C在69236步完成覆盖。共同后段CEM1679696次、MV1680304次。B严格保持原collection次数仍缺264304图；C按未见配额＋最小TV补偿，后段TV15.7352%达到下界。初版覆盖后直接sqrt重复的TV25.4387%记录单独保留，按无标签配额数学关系修正，不利用分割GT。

同一核心用于模拟和未来loader接口，未修改训练入口。新分支`codex/d2-coverage-design`；历史D12源码/配置/报告/权重/registry不修改。新增设计报告、核心原型、CPU模拟与测试、小型CSV/JSON。大ID计划和图位于workspace artifact。

实际验证：500个完整原sampler record对照和全480万索引来源计数一致；每策略336万后段ID均由同一核心查询；3rank×2worker进程重排/恢复/预取一致（每策略480个完整record，含69236阶段切换）；6项小池测试通过。全CPU单测51项，48通过、3项既有Windows不支持的POSIX测试跳过；registry一致性、仓库检查PASS（115 provenance、664历史文件、57实验记录）。正式GPU恢复仍未测试，不冒充训练已验收。

成本参照：新增70k SSL历史14.41h三GPU，新40k head约51min；C采样plan13.44MB、额外checkpoint状态190B。SQLite元数据估计CEM压缩读量比A高约2.57%，未把它当实测image-loader吞吐。SSL峰值磁盘约27.28GB，需另留head/预测空间。

GitHub第一次只读PR请求超时，重试成功。设计和记录提交推送新draft PR，以实际Git回执为准；不自动合并。推荐C待后续明确批准，B只模拟、A不重训。本轮完成后STOP。
