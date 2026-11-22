# MitoNet batch-four matched adaptation

| Split | Mix batch 4 | CEM batch 4 | Official | Mix batch 1 | CEM batch 1 | P0 | B100 | Barrier100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mv3 | 0.666098 | 0.516360 | 0.610209 | 0.175640 | 0.005337 | 0.795032 | 0.806347 | 0.821832 |
| cem_validation | 0.575983 | 0.617270 | 0.887468 | 0.371018 | 0.417479 | 0.497628 | 0.546517 | 0.624405 |
| mb1 | 0.503384 | 0.498871 | 0.712154 | 0.030906 | 0.146295 | 0.675877 | 0.668171 | 0.755627 |

Each arm took 40,000 optimizer steps with four crops per step and the MV-5 OneCycle schedule. Each Mix batch contained two MitoVerse crops and two CEM crops. The CEM-only arm used the same 80,000 CEM crops in the same slots and filled the other 80,000 slots with CEM draws. The two runs used seed 20261014 and the same official initialization.

Declared instance-supervised crop count was 140,156 for Mix and 160,000 for CEM-only. For mixed supervision, binary-only or ambiguous crops contributed to semantic/PointRend losses. Batch 4 versus batch 1 also increases the total number of crops per model at fixed optimizer steps and changes the batch sampling composition.

Per-image GT foreground/background counts match the released MitoNet baseline on all 4,057 evaluation units. Inference uses the MV-4 Engine2d configuration with scale 1, confidence 0.5 and TTA off. Each result uses one downstream seed. The official CEM exposure remains possible and MB-1 has been used for development.

## MB-1 empty-foreground predictions

- Official-MitoNet-v1 — 112 / 2,595
- MitoNet-Mix-B4 — 22 / 2,595
- MitoNet-CEM-B4 — 602 / 2,595

## Batch-four weight receipts

- MitoNet-Mix-B4 — `ff01204ad2165ad3c4467055ea25d84b628f9f248c4ef145c2136d53fa862008`
- MitoNet-CEM-B4 — `ca7c718265050415698311eeacc079fca7cf9815c1c638e426676ef9340ccbfa`
