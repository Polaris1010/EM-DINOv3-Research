# Paired raw EM and SAM MAM PH feasibility

424 label-free slices from all 212 volumes and 13 sources; each of 10 augmentations was applied to the same image before raw filtration and frozen SAM ViT-B feature extraction. MAM used OrgMIM's official affinity function (delta_v=0.5, delta_d=1.5; min of vertical and horizontal affinities). Both affinity and inverse boundary filtrations were tested. This is a ViT-B feasibility run; OrgMIM's README example uses ViT-H, so results should not be described as ViT-H replication.

Distances are approximate bottleneck values (e=0.001) after H0/H1 lifetime >=0.02 filtering. Equal sum is an exploratory display; it is not a tuned TopoAug score.

| Input | Augmentation | H0 p50 | H1 p50 | Sum p50 | Sum p90 |
|---|---|---:|---:|---:|---:|
| raw_em_paired | identity | 0.000 | 0.001 | 0.001 | 0.002 |
| raw_em_paired | flip_horizontal | 0.000 | 0.001 | 0.001 | 0.002 |
| raw_em_paired | flip_vertical | 0.000 | 0.001 | 0.001 | 0.002 |
| raw_em_paired | weak_jitter | 0.055 | 0.055 | 0.110 | 0.111 |
| raw_em_paired | strong_jitter | 0.203 | 0.170 | 0.376 | 0.447 |
| raw_em_paired | weak_blur | 0.023 | 0.018 | 0.040 | 0.106 |
| raw_em_paired | strong_blur | 0.141 | 0.095 | 0.239 | 0.411 |
| raw_em_paired | mild_crop | 0.102 | 0.074 | 0.186 | 0.262 |
| raw_em_paired | medium_crop | 0.126 | 0.098 | 0.239 | 0.346 |
| raw_em_paired | aggressive_crop | 0.156 | 0.122 | 0.290 | 0.403 |
| mam_affinity | identity | 0.001 | 0.001 | 0.001 | 0.002 |
| mam_affinity | flip_horizontal | 0.054 | 0.161 | 0.230 | 0.344 |
| mam_affinity | flip_vertical | 0.058 | 0.164 | 0.249 | 0.348 |
| mam_affinity | weak_jitter | 0.032 | 0.157 | 0.201 | 0.318 |
| mam_affinity | strong_jitter | 0.076 | 0.196 | 0.316 | 0.398 |
| mam_affinity | weak_blur | 0.049 | 0.167 | 0.248 | 0.345 |
| mam_affinity | strong_blur | 0.165 | 0.199 | 0.378 | 0.463 |
| mam_affinity | mild_crop | 0.065 | 0.213 | 0.314 | 0.384 |
| mam_affinity | medium_crop | 0.076 | 0.236 | 0.342 | 0.418 |
| mam_affinity | aggressive_crop | 0.091 | 0.278 | 0.368 | 0.461 |
| mam_boundary | identity | 0.000 | 0.001 | 0.001 | 0.001 |
| mam_boundary | flip_horizontal | 0.166 | 0.055 | 0.246 | 0.368 |
| mam_boundary | flip_vertical | 0.170 | 0.062 | 0.273 | 0.384 |
| mam_boundary | weak_jitter | 0.159 | 0.035 | 0.212 | 0.332 |
| mam_boundary | strong_jitter | 0.205 | 0.087 | 0.346 | 0.431 |
| mam_boundary | weak_blur | 0.174 | 0.053 | 0.306 | 0.367 |
| mam_boundary | strong_blur | 0.209 | 0.176 | 0.400 | 0.486 |
| mam_boundary | mild_crop | 0.234 | 0.070 | 0.332 | 0.413 |
| mam_boundary | medium_crop | 0.258 | 0.079 | 0.361 | 0.451 |
| mam_boundary | aggressive_crop | 0.301 | 0.096 | 0.389 | 0.501 |

## Paired ordering checks

Fractions below show the share of slices whose strong augmentation distance exceeded its mild counterpart; they measure dynamic range, not biological correctness.

| Input | Aggressive crop > mild crop | Strong blur > weak blur | Strong jitter > weak jitter |
|---|---:|---:|---:|
| raw_em_paired | 0.908 | 0.948 | 0.948 |
| mam_affinity | 0.797 | 0.880 | 0.950 |
| mam_boundary | 0.792 | 0.882 | 0.884 |

## Interpretation

SAM feature maps may change under flips because the encoder is not strictly flip equivariant. Intensity jitter changes filtration values even if geometry is unchanged. Crops change field of view. These distances alone cannot prove mitochondrial topology preservation or damage. Inspect source-stratified distributions and representative map images before selecting alpha, beta, or a threshold. No GT mask was read or used.

Figures and numeric summary are in this directory. The DINOv3 B_full training environment and GPU assignments were not changed.

## Verified finding and Stage 2 decision

The paired 424-slice comparison is complete. In the exploratory equal-weight H0+H1 distance, raw EM median horizontal-flip distance is 0.001, while SAM MAM is 0.230. MAM's aggressive-crop median is 0.368, and strong-blur median is 0.378. Aggressive crop exceeds mild crop in 79.7% of paired slices; strong blur exceeds weak blur in 88.0%. This shows a useful strength ordering, but a large MAM flip response. Source median MAM flip distance ranges from 0.136 to 0.313.

Do not set a production TopoAug threshold from this ViT-B MAM result. A strict distance gate would reject many harmless flips, and source-specific shifts are material. Next test flip canonicalization or compare MAM after applying the known geometric transform to the original map; then repeat with the intended prior (the OrgMIM README illustrates SAM ViT-H) and visual review. The present distances do not prove mitochondrial-specific topology, and no GT mask was used.
