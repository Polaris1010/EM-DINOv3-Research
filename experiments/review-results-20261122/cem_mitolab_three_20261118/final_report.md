# CEM-MitoLab matched frozen Primus comparison

Seed 20261014, 40000 updates. Equal collection or benchmark group means of image-level Dice. Empty GT and empty prediction score 1. Official MitoNet uses its released architecture.

| Evaluation | P0 | B100 | Barrier100 | B100 minus P0 | Barrier100 minus B100 |
|---|---:|---:|---:|---:|---:|
| cem_validation | 0.5507 | 0.5433 | 0.6408 | -0.0074 | 0.0975 |
| mb1 | 0.6729 | 0.7070 | 0.7560 | 0.0341 | 0.0490 |
| TEM-100 | 0.7928 | 0.7855 | 0.8540 | -0.0073 | 0.0685 |
| c_elegans | 0.4423 | 0.6326 | 0.6873 | 0.1903 | 0.0547 |
| fly_brain | 0.8334 | 0.8768 | 0.8880 | 0.0434 | 0.0112 |
| glycolytic_muscle | 0.8880 | 0.8851 | 0.8951 | -0.0030 | 0.0101 |
| hela_cell | 0.6943 | 0.7364 | 0.7685 | 0.0422 | 0.0320 |
| lucchi_pp | 0.8442 | 0.8882 | 0.9063 | 0.0439 | 0.0182 |
| salivary_gland | 0.2154 | 0.1444 | 0.2925 | -0.0710 | 0.1481 |

Official MitoNet historical MB-1 macro Dice: 0.7122.

The B100 to Barrier100 difference includes SSL data and sampling changes in addition to the barrier. MB-1 was previously exposed in method development. One downstream seed was used.
