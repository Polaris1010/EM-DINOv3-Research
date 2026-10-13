# MitoVerse Stage 2A2 interim result — 2026-10-02

## Decision state

- Stage 2A crop-PH gate remains archived FAIL. Its files were read only.
- Branch P is paused by the user pending a valid definition of the image augmentation measurement. No P calibration was run.
- Branch E is complete: a reusable frozen B_full probe was implemented and run on CMCC and CMCT.
- Branch S completed its 424 × 4 quantitative crop pilot. Its PASS/FAIL decision is pending the preregistered 100-pair human blind review.
- No 100k training or full MAM cache was started.

## Branch E: frozen B_full evaluation

The teacher ViT-L/16 backbone was loaded strictly from B_full step 100,000 and frozen. A fresh Primus multiscale decoder was trained with the same seed (20260907), fold-0 split, batch size 1, optimizer, loss, 3,000 steps, and validation protocol for each dataset. The validation head was selected with stride-20 Dice and assessed with full-volume, four-flip validation. A future checkpoint can use the same script and protocol.

| Dataset | Train/validation cases | Selected decoder step | Validation macro Dice | Micro Dice |
|---|---:|---:|---:|---:|
| CMCC | 27 / 3 | 2,000 | 0.8867 | 0.9160 |
| CMCT | 22 / 2 | 3,000 | 0.9615 | 0.9596 |

These are two small validation splits, not a comparison against the official natural-image initialization or an independent test set. They establish a reproducible B_full reference. Exact case metrics and hashes are in `B_full_CMCC_metrics.json` and `B_full_CMCT_metrics.json`.

## Branch S: spatial crop pilot

Directional SAM ViT-H affinities were extracted for the same 424 slices. Reconstructing the Stage 2 MAM from the directional maps matched the stored original maps exactly (maximum absolute difference 0). Four same-shape crop candidates per image were scored from actual boundary-crossing edges. The primary score was locked as mean crossing affinity (`cut_mean`, lower means less high-affinity connectivity cut). Persistent H0 scores were assessed as secondary diagnostics.

- Median within-image range of `cut_mean`: 0.0930 across 424 images; 99.1% had nonzero range. Every one of the 13 source groups had a positive median range, though several groups had only 2–10 images.
- Float16 and Gaussian affinity perturbation σ=0.001 both yielded median candidate-rank concordance 1.0 for `cut_mean`.
- Across 424 images and five flips/rotations, `cut_mean` differed by at most 1.8×10⁻⁷ over 8,475 valid comparisons.
- Persistent H0 excluding the unique largest component had median within-image range near 7.8 and was exactly invariant over 500 comparisons on 25 images after resolving tied largest components.
- Only 61.0% of the 1,696 crop boxes had all four crossing sides observable. Blind review pairs are restricted to candidates with four observable sides. Scores for boxes touching the context boundary need separate interpretation.
- Within-image candidate ranking correlation between `cut_mean` and the old PD crop score had median Spearman 0.0 (417 nondegenerate images). The old PD score is not used as a gate.

The 100 blind pairs include 50 OpenOrganelle and 50 spread across the other sources. Each pair uses the best and worst `cut_mean` candidate among those with four observable boundaries; A/B order is shuffled. The image panels contain only the shared EM image, SAM affinity overlay, and crop boxes. The answer key stays on the server and outside the review ZIP.

## Human review needed to decide S

Open `blind_panels_100.zip` and enter `A`, `B`, or `unclear` in `human_review_blank.csv` for each pair. Judge which crop cuts fewer visibly continuous structures. Do not use the score CSV while judging. On the judgeable subset, the locked gate is at least 70% agreement between the human preference and lower `cut_mean`. Unclear pairs are excluded from the denominator and reported. A visual audit is necessary because numeric range, source coverage, and invariance do not establish that the score captures the intended structure.

After the human review, decide S PASS/FAIL. Branch P remains paused until its measurement object is defined. No formal ViT-L long training is authorized by these interim results alone.
