# Historical source snapshots

These files preserve the exact code bytes used by existing experiments; `evidence/file_index.json` maps each copy to its real server/local origin and SHA256. The server experiment directories are authoritative execution locations. Relocation into this repository does not make these historical entry points directly runnable: their relative `R`, `src`, config, artifact and checkpoint paths refer to the original experiment layout. Do not invoke the launchers from this repository or silently rewrite those assumptions during bootstrap.

`pretraining/B100k` and `pretraining/B150k` preserve two distinct historical trainers. `segmentation/official_primus.py` is the actual MV-1 original readout; its AST matches the pinned upstream snapshot. The shared MedDINOv3 worktree and its modified Primus are distinct historical assets and remain on the server. `evaluation/` preserves the existing MV-2 no-TTA code, including new valid-mask scoring contracts; no inference was executed in this takeover.

The runnable portable utilities for this PR are in `scripts/`: evidence-based registry generation and CPU-only metadata/protocol checks. No upstream training framework is vendored or refactored. Import dependencies use the registered existing server environments; package freeze exports include editable and local build paths and are inventory records, not portable installation lockfiles.

Upstream references: DINOv3 commit `91ffb6803c74aeb6927209b82560142743763dd7`; MedDINOv3 Primus commit `a2460c2547899283446e520b84523454f7e25608`. Preserve upstream rights and consult the upstream license before any redistribution outside this private research scope.
