# CEM-MitoLab-only frozen Primus with batch size four

| Evaluation | P0 | B100 | Barrier100 | MitoNet-CEM-B4 | Official-MitoNet-v1 |
|---|---:|---:|---:|---:|---:|
| mv3 | 0.6122 | 0.6083 | 0.6305 | 0.5164 | 0.6102 |
| cem_validation | 0.6162 | 0.6026 | 0.6829 | 0.6173 | 0.8875 |
| mb1 | 0.6923 | 0.6976 | 0.7453 | 0.4989 | 0.7122 |

The three frozen encoders used the original Primus head, identical initial head state and seed 20261014. Each model completed 40,000 optimizer steps with four CEM crops per update. The ordered 160,000 crops match the batch-four MitoNet CEM-only arm. AdamW and the original Primus learning-rate schedule were retained. Native-grid inference uses the fixed 40k checkpoint, windows of 256, stride 128, threshold above 0.5 and TTA off.

MV3 covers 38 regions and 12 sources. CEM covers 1,424 images in 30 held-out collections. MB-1 covers 2,595 images or slices in seven groups. All 4,057 evaluation units were matched by image ID and ground-truth foreground/background counts. CEM and MB-1 batch-one scores are included for comparison. Four times the crop exposure and a changed sampler accompany the batch-size change at the fixed update budget.

The official CEM scores may reflect upstream supervised exposure. MB-1 is an exposed diagnostic. This evaluation uses one decoder seed.

| Model | Encoder SHA256 | Head SHA256 |
|---|---|---|
| P0 | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` | `8951a410c8b01707729f90a4c5457a4ba3cb10b08f2156bd05f0ca0d71df2461` |
| B100 | `9603877244ba40f3cbeccdea18d69b0d3b818dc0fc6b09f839fd6d4189ce2de5` | `c10e5967614e8f08b59aa2cfbf9531e2ef7f04362149afe0485aff06904d0b5f` |
| Barrier100 | `9188d26f90216ee966fb8f80fdfde9d33d685202b35e62fa78b34b5b41c073a4` | `1e0ca4324aa5dd3e939d4683da8a732b916ca1fb2119db9cac1bf91e0836d373` |
