# Full MitoVerse restart execution record

Date: 2026-09-28 UTC. Server: `root@192.168.203.97`.

Run: `/opt/data/3_117/tanghr/codex/mitoverse_full_restart_20260928_133515`
Service: `mitoverse-full-seed0.service`.

## Verified before launch

- The six old v2 watcher, training, and evaluation services were inactive with
  MainPID 0. No old training or evaluation process was active.
- Catalog and on-disk Zarr directories each contained 212 volumes from 13
  sources. The new manifest path set and IDs exactly matched both. It contains
  31 old train, 8 old validation, 16 old legacy_dev_test, and 157 old excluded
  volumes, including all 157 OpenOrganelle volumes. The old split is only a
  `legacy_split` trace field.
- Manifest SHA256:
  `e25c88f86d79116d9eef21f733328ed0d5f4518c25ac1061f910684bb0c36fa0`.
  All 212 centers and one 512 × 512 padded image crop per volume were readable.
- Official DINOv3 ViT-L/16 weight SHA256 matched
  `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`.
- Source-wide image-only topology calibration completed at 32 crops per source.
  Near-identity candidates were excluded before locking T/LT selection to the
  new manifest hash.
- B/L/T/LT each completed 10-step smoke and 100-step stability runs from the
  official weight, with finite loss and gradients, parameter movement, and
  matched sample chains. A new-run step-10 resume and continuous step-20 B
  run had identical per-step sample chains. Technical-test weights are not
  formal initializers.
- `evidence/formal_seed0_admission.json` passed, including full-manifest
  revalidation, parsed configuration checks, method hashes, official weight
  hash, and the recovery check.
- At launch, GPU1/2 carried only the website process, GPU3 was free, the
  website returned HTTP 200, and no STOP marker or formal job existed.

## Formal launch

At approximately 2026-09-28 13:55 UTC, a new transient systemd service started
`coordinator_formal_seed0.py`. B/L/T were dispatched to physical GPU1/2/3;
LT is queued for the first free card. The actual worker commands have no
`--resume` argument. All three `initialization.json` files record official
natural ViT-L/16, strict load, missing/unexpected keys empty, step 0, and the
212-volume manifest. Early progress files showed real optimizer steps
incrementing on all three GPUs, with all 13 sources sampled. By step 670,
each of B/L/T had sampled all 212 volumes at least once, with identical sample
chains. The old watcher and evaluation queue were not restarted.

At step 1,000, B/L/T each wrote a 5.6 GB complete checkpoint. The files were
read back and their actual SHA256 hashes matched their atomic receipts; the
verification is on the server at `evidence/first_checkpoint_verification.json`.
After that check, B/L reached step 1,200 and T step 1,190. The service
remained active, no STOP marker appeared, and the website returned HTTP 200.

This record describes a running pretraining experiment. The 100k-step arms,
their milestone checkpoints, downstream supervised heads, and final paper
metrics are not yet complete. Full-MitoVerse supervised training requires a
separate label and valid-region audit; in-sample MitoVerse evaluation cannot
be presented as independent generalization.
