# Full-MitoVerse Stage 1 execution status (2026-09-29)

Remote root: `/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120`.

The old B/L/T full-restart service was stopped after checking process identities. Its checkpoints remain in the old run and were not used here. LT was never dispatched.

The new manifest contains 212 volumes from 13 sources and 48,722 legal Z slices. Catalog and on-disk Zarr path sets matched. All 636 representative first/middle/last image reads passed. The label-free uniform slice sampler passed 1,000 augmented samples. Strict official natural DINOv3 ViT-L/16 initialization matched 368 backbone keys; official inference and training student CLS and patch outputs matched exactly. Single-GPU, three-GPU, and same-run resume smoke tests passed. The formal admission passed 869 gates.

`B_full` is running as `mitoverse-stage1-B-full-v2.service` on physical GPUs 1, 2, 3, with GPU 0 forbidden and the website preserved. At the last report write it was at step 1,090 / 100,000. The first formal attempt failed at step 0 because NFS file timestamp skew triggered a false stale-heartbeat check; that attempt remains on disk. Attempt1 restarted from the official weights after the heartbeat logic was corrected and retested. The step-1,000 full optimizer checkpoint completed atomically at `runs/B_full/formal_attempt1/checkpoints/step000001000/full.pth`, 5,607,208,357 bytes, SHA256 `39727b61d8a05c47381b89b621e7a51462021660956b4367272726f46c00c32b`. Step 1,060 was observed after that checkpoint, with website HTTP 200 and service active. This is an interim status; 100,000-step training is not complete.

Topology raw-EM PH analysis completed on 424 label-free slices (2 for every volume), 10 augmentations, and 13 source groups, yielding 4,240 rows, H0/H1 boxplots, a source heatmap, and a report under `topology_feasibility/`. GUDHI is isolated under that directory's `vendor/`. OrgMIM MAM code/checkpoint were unavailable on the server and GitHub was unreachable from it. The raw-vs-MAM comparison, a defensible topology threshold, and complete Stage 1 acceptance remain pending. No GT masks were used.

Remote reports: `reports/B_training_report.md`, `reports/stage1_summary.md`, `topology_feasibility/report.md`; source and audit commit `ab91ea2af83e3067bb82c1bff9c5afd463ba498c` on `codex/stage1`. The guard checks GPU ownership, website health, disk, and heartbeat while the systemd training service remains active.
