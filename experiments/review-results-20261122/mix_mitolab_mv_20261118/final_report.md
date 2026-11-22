# Mixed MitoVerse and CEM-MitoLab supervision

Four trained systems used exactly the same 40,000 sampled crops, seed 20261014, 20,000 from each domain. A fifth system retained the official released MitoNet weights. The image shapes, image IDs, and foreground/background counts match for all five.

| Split | P0 | B100 | Barrier100 | MitoNet-Mix | Official unchanged | Mix minus official |
|---|---:|---:|---:|---:|---:|---:|
| cem_validation | 0.4976 | 0.5465 | 0.6244 | 0.3710 | 0.8875 | -0.5165 |
| mv3 | 0.7950 | 0.8063 | 0.8218 | 0.1756 | 0.6102 | -0.4346 |
| mb1 | 0.6759 | 0.6682 | 0.7556 | 0.0309 | 0.7122 | -0.6812 |

MV3 pools confusion counts per region then averages regions per source and weights 12 sources equally. CEM and MB-1 average image-level metrics per collection or dataset then weight groups equally. Empty/empty Dice is 1. The official CEM and MB-1 scores came from the matched rerun; MV3 reuses the same released model from MV4. CEM supervision exposure is possible for the released model. MB-1 participated in development.

Primus uses a frozen encoder and fresh head. MitoNet-Mix updates all weights using the official MV5 optimizer and OneCycle schedule with batch size one to match sample exposure. Reliable instance labels supervise center and offset terms; ambiguous binary MV crops supervise semantic and PointRend terms only. The initialization and optimizer settings differ by model family. Single downstream seed.

Per-group scores and both deltas are in dice_comparison.csv. Within-encoder mixed versus CEM-only comparisons are in mixed_vs_cem_only.csv.
