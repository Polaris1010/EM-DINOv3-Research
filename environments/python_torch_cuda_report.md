# Python / PyTorch / CUDA report

This report is a human-readable index of the read-only 2026-10-13 environment registry. Exact executable paths, package versions and module origins are in `environment_registry.json`; actual import checks are in `../evidence/import_check.json`.

| Environment | Python | Torch import | torch.version.cuda |
| --- | --- | --- | --- |
| dinov3_shared | 3.11.14 | 2.10.0+cu128 | 12.8 |
| meddinov3 | 3.11.9 | 2.5.1 | 12.4 |
| mitoverse | 3.11.15 | NOT_INSTALLED | NOT_APPLICABLE |

The GPU driver reports CUDA 12.2 via nvidia-smi. Package metadata, successful CPU imports and observed historical/current runtime evidence are separate claims; no new GPU workload was used for this inventory. Full optimizer checkpoint safe loading encounters a numpy serialization restriction; teacher/student exports and all nine MV-1 head dictionaries load with weights_only=True on CPU.

Additional executables are registered only as discovered; dinov2 import timed out and the remaining optional environments were not imported. No package changes were made. Freeze records include editable/local paths and are not portable install specifications.
