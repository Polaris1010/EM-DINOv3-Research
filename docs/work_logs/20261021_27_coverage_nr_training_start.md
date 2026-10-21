# Coverage-NR启动

用户明确授权开始训练。本轮单组Coverage-NR，P0新初始化、连续100k、固定现有两池、global48/seed0、DINO+iBOT+0.1KoLeo。之后首seed原版冻结Primus40k/38dev/七组MB1，结束停止，不自动额外组或新loss。

新代码src/coverage_nr复用实际已跑通的D12训练器/loader/readout，原文件不改。全程配额与无放回队列，实际同一Plan.choose供CPU验证与loader。480万domain位置、全部1425433 CEM ID覆盖和432完整位置分rank/worker重排/恢复已通过；计划SHA bf56bad49c74eb484f0eee125370f31c386379c56b8ea877a8467fc4685d500e。计划覆盖不等同实际已访问覆盖。

现场preflight：旧D1/D2 complete；GPU1/2各有网站进程约464MiB，保留，空闲显存充分；GPU3空闲，GPU0不调用；网站200。独立root coverage_nr_p0_20261021已部署，smoke10步新初始化完成、接续至12步检查中，以服务器收据为准。尚不把正式100k写成已完成/已启动。

大ID计划和checkpoint留服务器/本地artifact；Git只保存代码、小型计划metadata、CPU验证和Markdown。新的registry条目通过生成器追加，旧57条记录保持。完整执行/交接见docs/Coverage_NR_execution.md。

GPU smoke已实际PASS：新P0完成10步，全student/EMA/optimizer/loss-centers/3rank RNG与同计划恢复至12步；三rank参数最大差0，student/teacher均发生更新。收据evidence/Coverage-NR/smoke.json。51项CPU单测48通过、3项既有POSIX跳过；registry59条及repository validation PASS。正式训练尚待提交代码后启动，后续收据会追加。
