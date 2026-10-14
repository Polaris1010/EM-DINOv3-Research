# Official upstream definitions

These unchanged official empanada-napari source snapshots provide `Engine2d`, `Preprocessor`, normalization and local TorchScript loading. The headless adapter compiles only those definitions; Qt/GUI imports and module-level side effects are not executed. Runtime panoptic rendering, tiling, connected components and tile consensus come from the official `empanada-dl==0.1.7` PyPI wheel.

Sources and full-file SHA256 are recorded in `evidence/MV-4/official_assets.json`. Upstream Git commit is UNVERIFIED for these raw snapshots; content hashes, official origin and actual wheel version are recorded instead of inventing a commit. The model configuration is the unmodified official MitoNet_v1 configuration.

Original project: [volume-em/empanada-napari](https://github.com/volume-em/empanada-napari), maintained by Ryan Conrad, Kedar Narayan and contributors, distributed under [BSD-3-Clause](https://github.com/volume-em/empanada-napari/blob/main/LICENSE). The official PyPI backend wheel includes its own license. Model weights are released separately at [Zenodo6861565](https://zenodo.org/records/6861565), under CC-BY4.0, and are never included in Git.
