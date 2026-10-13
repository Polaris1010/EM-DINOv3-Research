# Stage 1 transition preflight (2026-09-29)

## Current execution boundary

SSH access was restored during this session. The new run directory is
`/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120`.
Preparation is underway; `B_full` training has **not** started.

## Existing run that must be reconciled first

The local [execution record](mitoverse_full_restart_control/execution_record.md)
identified `/opt/data/3_117/tanghr/codex/mitoverse_full_restart_20260928_133515`.
Live checks found its B/L/T workers around steps 11,330/11,330/10,800. Their
registered identities were checked, `STOP.json` was written, and only those
worker process groups received SIGTERM. The service is now inactive with
MainPID 0; LT was never dispatched. The latest preserved checkpoint receipts
are B=11,000, L=11,000, T=10,000. GPU1/2 returned to the website-only 464 MiB
state, GPU3 is free, and the website returned HTTP 200. The old run remains
historical and cannot initialize `B_full`.

## New Stage 1 observations

- Catalog and local Zarr path sets matched: 212 volumes, 13 sources, including
  157 OpenOrganelle volumes.
- The complete legal Z-slice pool has 48,722 slices. The new manifest hash is
  `3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9`.
- First, middle, and last image planes read successfully for all 212 volumes:
  636 reads, no failures. This is not a full chunk checksum.
- The live official weight hash matched the requested
  `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`.

## Confirmed local protocol gaps

- The existing full manifest is a top-level list. Stage 1 requires an object
  with version, catalog provenance, counts, role, and `volumes`.
- Existing rows record `image_shape_zyx`, `image_dtype`, and `axes`, but not the
  requested explicit image array path, chunks, and voxel resolution fields.
- The existing SSL sampler selects source uniformly, then volume uniformly;
  Stage 1 requires uniform sampling over the complete legal slice pool.
- The existing manifest audit reads one center voxel per volume. Stage 1
  requires first, middle, and last representative image slices, with errors
  attributed to volume and slice.
- The existing B trainer is a custom single-GPU runner merged with an older
  continual config. Stage 1 requires a fresh configuration derived from the
  official DINOv3 SSL defaults, a documented diff, backbone output equivalence,
  single-GPU and distributed smoke tests, and a three-GPU formal B run.
- The existing topology calibration is image-only, but does not establish the
  requested raw EM versus OrgMIM MAM persistent-homology comparison.

## Remote sequence after access is restored

1. Capture live service, process, GPU, disk, and website state. Identify the
   current experiment's worker PIDs and checkpoint receipts.
2. Stop only conflicting L/T/LT work using its own coordinator's graceful
   mechanism, preserving current checkpoints. Keep GPU 0 and website workers
   untouched. Record the resulting state.
3. Create a new timestamped `mitoverse_full_stage1_*` directory. Rebuild and
   audit the catalog-to-Zarr manifest from live data, without copying the old
   manifest or inheriting its sampler.
4. Complete the official-weight, reader, label-access, initialization,
   configuration, and smoke gates. Launch 100k B only after every gate passes.
5. Complete topology feasibility separately and report raw/MAM results only
   when actually measured.

No count, hash, process state, or training result from the 2026-09-28 record
should be presented as a current Stage 1 observation.
