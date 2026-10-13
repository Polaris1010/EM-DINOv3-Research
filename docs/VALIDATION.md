# 初始化验证

本轮通过SSH只读盘点指定服务器。catalog与Zarr store计数均212，网站HEAD HTTP200。PyTorch、MONAI、dynamic-network-architectures、nnunetv2、实验实际original Primus可在meddinov3 CPU环境导入；dinov3_shared torch也可导入。额外dinov2环境检查超时已登记。

九个MV-1 head均存在，约178MB/个，CPU weights_only/mmap读取通过，存储step=20000、44576514参数、model/seed/encoder SHA与对应运行记录一致。P0、B100k/B150k teacher/student可安全读取；full optimizer状态的numpy序列化在安全模式下被拒绝，保留错误metadata记录，不改为不受限加载。

离线验证检查精确text快照SHA、源码AST一致性、26条注册记录的证据和必要字段、九个head关联、同seed初始decoder与采样序列匹配、监督train/dev分离、reserved test排除、原推理设置，以及可能的凭据/大文件误入。执行Python源码compile不运行其入口；检查无需torch依赖或GPU。

实际本地运行回执在 `evidence/local_validation.json`：注册表生成一致、来源/metadata/protocol校验、五项负面/正常测试、Git diff检查全部通过，另核验115份快照的staged blob SHA未被Git换行转换。测试分别检查真实关联、错误encoder hash、head换位、TTA漂移、reserved test混入。

远程GitHub Actions已运行，commit `0f290b9231e168a59d5b8e8c8e86f1a88f3b8505` 的push/PR两次CI均success。具体已核验run链接见 `docs/GITHUB.md`；每次新提交仍需看其自己的Checks，不能用本地结果或旧commit代替。

未进行全部Zarr chunks checksum、权重大文件完整重新哈希、全部环境导入、GPU训练/推理smoke或严格acquisition重新审计。这些边界不改变已有结果，也不阻塞当前定义的开发评测。

`.gitattributes`关闭text换行转换，确保快照字节SHA在Windows/Linux间保持一致；历史src/config/experiment/Primus证据及原始server_health输出保留trailing whitespace，Git whitespace检查对这些快照设置例外。来源索引覆盖的快照由字节哈希检查保护，新撰写文档与工具正常执行diff检查。
