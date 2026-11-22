# C100 three matched decoder seeds

| Model | MV3 Dice mean ± SD | MB1 Dice mean ± SD | Muscle Dice mean ± SD |
|---|---:|---:|---:|
| C0_Vanilla_100k | 0.840090 ± 0.001485 | 0.472593 ± 0.011023 | 0.232078 ± 0.041631 |
| C1_Barrier_100k | 0.840019 ± 0.000847 | 0.525780 ± 0.008315 | 0.526469 ± 0.025690 |
| Trajectory-Lerp05 | 0.838481 ± 0.001309 | 0.492458 ± 0.015660 | 0.444851 ± 0.031052 |

Fixed encoders; one SSL trajectory; sample SD across three decoder seeds. MB1 is an exposed diagnostic. No new SSL, probe training, threshold/scale search or further experiment.
