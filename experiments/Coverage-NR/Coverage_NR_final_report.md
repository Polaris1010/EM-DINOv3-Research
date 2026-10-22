# Coverage-NR P0→100k final report

One exploratory full-coverage run; historical D2 differs in collection distribution and LR history. No causal coverage-only claim. Fixed frozen Primus seed20261014,40k,156train/38dev,TTA OFF. MB1 is a previously exposed diagnostic, not independent blind testing.

Development source-equal metrics: {"dice": 0.8318207975761698, "precision": 0.8461748432004582, "recall": 0.8415175235999802, "hd95_nm": 505.28240758, "nsd_16nm": 0.7432368821605588, "predicted_foreground_fraction": 0.08045469072653039}
MB1 seven-dataset equal Dice: 0.44002743325886123
Consumed ID coverage: {"unique_cem_ids": 1425433, "coverage": 1.0, "cem_draws": 2398767, "mv_draws": 2401233}

| Source | NR Dice | historical D2 | difference |
|---|---:|---:|---:|
| casser20 | 0.936849749852063 | 0.9367920735248528 | 5.767632721020988e-05 |
| guay21 | 0.7489866874478357 | 0.7627149515426673 | -0.013728264094831633 |
| haberl18 | 0.5331487615563356 | 0.5408911943003563 | -0.007742432744020644 |
| han24 | 0.7006578287374169 | 0.6846355456883041 | 0.016022283049112795 |
| jiang25 | 0.8807869523785925 | 0.8831616539612342 | -0.0023747015826417472 |
| kunduri22 | 0.9562748621285431 | 0.9514038188798742 | 0.004871043248668827 |
| mekuc20 | 0.9343374591900482 | 0.9315534554046072 | 0.0027840037854409694 |
| muller21 | 0.896010170255661 | 0.8965818328996689 | -0.0005716626440078754 |
| openorganelle | 0.7089692304476387 | 0.8206498369687227 | -0.11168060652108402 |
| turner22 | 0.8231559732903247 | 0.8281567777834585 | -0.005000804493133781 |
| wei20 | 0.9312695582996166 | 0.9286225391688205 | 0.0026470191307960134 |
| xiao18 | 0.9314023373299614 | 0.931380910163304 | 2.1427166657317365e-05 |

Per-region/source and MB1 per-image/dataset metrics, checkpoint SHA256 and immutable sampling plan accompany this report. Inspect per-source regressions and Glycolytic muscle before judging value; ID coverage is not structure coverage. No unseen-acquisition claim or automatic D3/longer training. This authorized run ends here.
