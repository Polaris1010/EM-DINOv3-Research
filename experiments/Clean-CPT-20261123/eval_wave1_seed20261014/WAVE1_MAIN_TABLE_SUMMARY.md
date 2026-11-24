# Clean-CPT Wave 1 main table

All six heads used seed 20261014, batch 4 and 40,000 steps. Postprocessing used 175 CEM-val views only. Main MB-1 metrics use an equal mean of seven groups. The released native MitoNet reference for MB-1, CEM-val and HQ is from the benchmark's matching evaluation.

| Backbone | Head | Dice | IoU | Interior miss r3 | PQ | AP50 | AP75 | MAE | Muscle Dice | Muscle PQ | CEM-val PQ | HQ square holes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | Head C | 0.75034 | 0.65152 | 0.09481 | 0.50614 | 0.48769 | 0.35863 | 4.47288 | 0.89584 | 0.48853 | 0.59004 | 0 |
| P0 | Head D | 0.73279 | 0.63110 | 0.09066 | 0.49110 | 0.46047 | 0.35624 | 7.78550 | 0.88569 | 0.46421 | 0.59597 | 0 |
| Clean-C0_100k | Head C | 0.77143 | 0.67147 | 0.07360 | 0.49930 | 0.46877 | 0.33845 | 5.18300 | 0.86134 | 0.44060 | 0.51761 | 0 |
| Clean-C0_100k | Head D | 0.76398 | 0.66205 | 0.06699 | 0.51832 | 0.48901 | 0.38482 | 7.11659 | 0.80102 | 0.39824 | 0.55033 | 0 |
| Clean-C1_100k | Head C | 0.75807 | 0.66235 | 0.08288 | 0.50907 | 0.49442 | 0.36096 | 3.98978 | 0.85658 | 0.41516 | 0.56772 | 0 |
| Clean-C1_100k | Head D | 0.75960 | 0.66256 | 0.07699 | 0.51340 | 0.48917 | 0.37695 | 6.50443 | 0.84783 | 0.40218 | 0.55628 | 0 |
| Official-MitoNet-v1 2D | Native | 0.71215 | 0.61551 | 0.09836 | 0.46658 | 0.44375 | 0.33418 | 6.77082 | 0.90502 | 0.45101 | 0.76096 | 1 |

## External local MitoEM 2.0 overlay

The test summary gives each scored volume equal weight. Han24 zeros outside its annotated ROI are ignored. These volumes are excluded by exact identifier from Clean-CPT's registered MitoVerse training manifest. This does not establish P0 or all CEM image provenance. No local official CellMap test GT was verified.

| Backbone | Head | Split | Volumes | Scored planes | Dice | PQ |
|---|---|---|---:|---:|---:|---:|
| P0 | Head C | test | 10 | 5102 | 0.69475 | 0.44517 |
| P0 | Head C | val | 1 | 100 | 0.80849 | 0.46916 |
| P0 | Head D | test | 10 | 5102 | 0.69735 | 0.45897 |
| P0 | Head D | val | 1 | 100 | 0.81465 | 0.50965 |
| Clean-C0_100k | Head C | test | 10 | 5102 | 0.79680 | 0.48974 |
| Clean-C0_100k | Head C | val | 1 | 100 | 0.81352 | 0.46434 |
| Clean-C0_100k | Head D | test | 10 | 5102 | 0.78146 | 0.52968 |
| Clean-C0_100k | Head D | val | 1 | 100 | 0.82139 | 0.51724 |
| Clean-C1_100k | Head C | test | 10 | 5102 | 0.78490 | 0.49457 |
| Clean-C1_100k | Head C | val | 1 | 100 | 0.82167 | 0.49645 |
| Clean-C1_100k | Head D | test | 10 | 5102 | 0.77664 | 0.52761 |
| Clean-C1_100k | Head D | val | 1 | 100 | 0.83227 | 0.52854 |
| Official-MitoNet-v1 2D | Native | test | 10 | 5102 | 0.55172 | 0.34293 |
| Official-MitoNet-v1 2D | Native | val | 1 | 100 | 0.82681 | 0.47328 |
