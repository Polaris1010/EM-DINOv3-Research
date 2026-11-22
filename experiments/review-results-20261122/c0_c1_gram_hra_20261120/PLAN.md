# C0/C1 matched continuation
- Both branches use the original image-only D2 mixture and exposure history, optimizer and per-rank RNG restored.
- C0 and C1 each add 20k Gram updates then 10k larger-crop updates. No new Barrier term after 100k.
- Gram uses the upstream loss, weight 1, same-view 256 student / 512 Gram teacher, branch EMA anchor at 100k and refresh at 110k.
- HRA uses a budget-adapted single group from the official HRA recipe, 512 global / 112 local / 768 Gram teacher, initialized with each 120k EMA anchor and kept fixed in this phase.
- Fixed LR 1e-6 inherited from the terminal 100k LR. WD 0.04; batch 48 across physical GPUs 1,2,3. Queue gates HRA on a completed Gram receipt. New runs live in this directory.
- MB-1 is the labeled diagnostic and HQ-5nm is the qualitative unseen-domain check.

HRA transition update. Probe 768 global / 336 local / 1152 Gram, then 768 / 112 / 1152, then 512 / 112 / 768. Trials run 12 disposable complete updates at the original batch of 48 and require at least 1 GiB estimated peak headroom on each allowed GPU. The first passing geometry is shared between C0 and C1 and formal training restarts at 120k. Probe steps are excluded from training budgets. A non-memory failure stops the stage for diagnosis.
