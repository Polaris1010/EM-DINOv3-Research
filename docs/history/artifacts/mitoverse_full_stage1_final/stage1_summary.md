# Full-MitoVerse Stage 1 final execution report

## Current facts

The catalog and on-disk Zarr path sets matched exactly. The fresh manifest has **212 volumes** from **13 sources**, including **157 OpenOrganelle volumes**, and **48,722 legal Z slices**. All 636 first/middle/last representative image reads passed; no unreadable volume was found in this audit and no repair was required. This is a representative read audit, not an all-chunk checksum. Manifest SHA256: `3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9`. Sampling is uniform over legal slices, with no old split or label filter.

## B_full

Initialization was the official natural-image DINOv3 ViT-L/16 weight, SHA256 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`; DINOv3 source commit `91ffb6803c74aeb6927209b82560142743763dd7`. All 368 backbone keys matched, with zero feature difference for checked CLS and patch outputs. **No old P1/P2/B/L/T/LT checkpoint or optimizer was loaded.** The successful formal attempt started at step 0 with no resume; an earlier attempt failed at step 0 due to an NFS heartbeat timestamp issue and was preserved separately.

`B_full` completed **100,000/100,000 optimizer steps** on physical GPUs 1–3. The global batch was 48; total wall time was 22.25 hours and mean end-to-end time was 0.801 seconds per step, including checkpoint overhead. Peak allocated GPU memory reported by PyTorch was 9.54 GiB per rank (live nvidia-smi usage during training was about 11 GiB per card). The objective was DINO=1, iBOT=1, KoLeo=.1, with Gram, LAST, topology, and segmentation branches disabled.

Three rank logs each contain 100,000 continuous steps with zero nonfinite loss/gradient/schedule scalar and zero step gaps. Each rank's first-sample records covered all 212 volumes. DINO+iBOT+KoLeo loss decreased from 16.625 at step 1 to the last rank-0 loss 12.5625; student and teacher tracked parameters moved, and final tracked inter-rank parameter difference was 0.0. Loss stability does not establish representation improvement. The final full optimizer checkpoint is `runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth`, SHA256 `12bbf09ffcd7f93cbad88c4e55df55d36b3a299990ad976bf29af51a118a265b`; student and teacher backbone exports exist beside it. Training service exited successfully; the website stayed available. Formal admission passed 869 gates.

## Topology feasibility and Stage 2 gate

Raw EM PH and a frozen Meta SAM ViT-B/OrgMIM-affinity MAM were tested on two label-free slices from every volume: 424 images, 10 augmentations, 13 sources, and 12,720 paired H0/H1 distance rows across raw, affinity, and inverse-boundary filtrations. No GT mask was used. The MAM test is ViT-B feasibility; the OrgMIM example uses SAM ViT-H. Code reference: https://github.com/yanchaoz/OrgMIM and TopoCL topology reference: https://github.com/gm3g11/TopoCL .

For the exploratory equal-weight H0+H1 bottleneck distance, median horizontal-flip distance was **0.001 for raw EM** and **0.230 for MAM**. MAM median aggressive-crop and strong-blur distances were **0.368** and **0.378**. Strong crop exceeded mild crop in 79.7% of paired images; strong blur exceeded weak blur in 88.0%. The MAM flip median varied by source from 0.136 to 0.313. Raw grayscale PH is not a trustworthy proxy for mitochondrial-specific topology; MAM has useful strength ordering but is **not more stable under flips**. Neither result proves biological topology preservation.

**No absolute TopoAug threshold is accepted for Stage 2.** A current distance gate would reject many harmless flips. First test a transform-aligned MAM comparison and review representative membranes across sources, ideally with the intended SAM prior. After those checks, a defensible starting search is a source-normalized benign-distance quantile in the **85th–95th percentile** range, then assess aggressive-crop false acceptance. This is a calibration proposal, not a selected threshold. Do not launch formal T1 until the flip and source-dependence gate passes. See `topology_feasibility/mam_sam_vit_b/paired/report.md` for full distributions.

## Reproducibility and scope

The resolved config, environment snapshot, source commit and patch, manifest, admission, step logs, checkpoints, and PH metrics are stored in this separate run. Old experiments remain preserved. No LAST/LT/T long run, segmentation fine-tuning, MitoNet evaluation, or frozen probe was started. Baseline training is complete; topology threshold readiness remains blocked by the observed MAM flip response.
