# 实际环境

本轮未运行pip install/upgrade/uninstall。详见 `environments/environment_registry.json`、freeze文本及 `evidence/import_check.json`。

| 环境 | Python | PyTorch / 编译CUDA | 关键依赖与核验 |
| --- | --- | --- | --- |
| dinov3_shared | 3.11.14 | 2.10.0+cu128 / 12.8 | torch导入成功；nnUNet/MONAI/dynamic-network-architectures/zarr未安装在该环境，历史SSL reader有单独deps |
| meddinov3 | 3.11.9 | 2.5.1 / 12.4 | nnunetv2 2.5.2、MONAI1.5.0、dynamic-network-architectures0.4.2、zarr3.1.6；torch/MONAI/nnunet/Primus实际CPU导入成功 |
| mitoverse | 3.11.15 | 无torch | zarr2.18.7；只用于确认存在和package metadata，不认定其为MV-1环境 |

nnunetv2为namespace package，实际路径包括meddinov3 site-packages和 `/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet/src/nnunetv2`。MV-1实际Primus导入为独立实验 `mv_generalist_stage1_20261013/src/official_primus.py`；不能因为共享nnUNet仓库存在modified primus就认定本轮用了它。

MV-1已经结束，inactive transient service不再保留可用ExecStart；已完成launcher源码明确使用meddinov3，实际metric日志warning路径也指向该环境。MV-2活动service和当前inference进程明确使用同一executable。注册表附带对应证据强度，不仅根据环境名称作判断。

额外发现dinov2、swinunetr、sam3、microsam、medsam、miniconda3 executable；其中dinov2只读导入检查40秒超时，余者未进一步导入。只登记真实路径，不创建新环境。不影响当前已验证的Primus评测。未发现并登记的独立TopoCL/empanada Python环境不应被虚构。

服务器nvidia-smi驱动535.171.04显示CUDA12.2；这是驱动报告，不能与torch编译CUDA版本或GPU执行兼容性混为一谈。现有训练/推理资产有实际执行记录；本轮CPU导入与safe metadata读取没有验证新GPU kernel兼容性，也不修改驱动/环境。

freeze包含本地构建路径、editable仓库引用及既有框架commit，是盘点证据，不是跨机器pip可直接安装的portable lockfile。后续需要迁移时单独制定环境重建任务，不在初始化中升级包。
