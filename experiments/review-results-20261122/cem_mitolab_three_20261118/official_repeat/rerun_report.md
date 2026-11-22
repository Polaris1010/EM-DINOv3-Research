# Official MitoNet fresh inference

Seed 20261014, official Engine2d, scale 1, confidence 0.5, TTA off. Frozen released checkpoint SHA256 487d63dd1deb971a3e374785b350673ee5427e0ffcd44880bf9a54f6c2c4bf04.

| Dataset | P0 | B100 | Barrier100 | Official MitoNet rerun | Barrier minus MitoNet |
|---|---:|---:|---:|---:|---:|
| cem_validation | 0.550714 | 0.543316 | 0.640775 | 0.887468 | -0.246693 |
| mb1 | 0.672922 | 0.707006 | 0.755966 | 0.712154 | 0.043812 |
| TEM-100 | 0.792809 | 0.785503 | 0.854048 | 0.818267 | 0.035780 |
| c_elegans | 0.442275 | 0.632569 | 0.687256 | 0.610827 | 0.076429 |
| fly_brain | 0.833429 | 0.876833 | 0.888023 | 0.899098 | -0.011075 |
| glycolytic_muscle | 0.888050 | 0.885092 | 0.895142 | 0.905023 | -0.009881 |
| hela_cell | 0.694261 | 0.736432 | 0.768472 | 0.733706 | 0.034766 |
| lucchi_pp | 0.844242 | 0.888175 | 0.906328 | 0.890342 | 0.015986 |
| salivary_gland | 0.215388 | 0.144437 | 0.292490 | 0.127812 | 0.164678 |

Maximum absolute group Dice difference from archived rounded MB-1 metrics: 4.79782361e-08.
Identical image IDs, per-image foreground and background counts verified across all four systems. CEM holdout split was defined for the three new heads. Original MitoNet supervised exposure to these images is possible. The inference seed does not alter its historical training seed.
