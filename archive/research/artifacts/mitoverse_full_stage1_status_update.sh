set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
systemctl reset-failed mitoverse-full-seed0.service
cat > "$run/STATUS.json" <<'EOF'
{
  "stage": "preparation",
  "B_full": "not_started",
  "topology_feasibility": "not_started",
  "formal_admission": "pending",
  "manifest": "data/pretrain_all_mitoverse.json",
  "manifest_sha256": "3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9",
  "volume_count": 212,
  "source_count": 13,
  "slice_pool_count": 48722,
  "reader_audit": "pass: 636 representative image slice reads",
  "prior_full_restart": "stopped; checkpoint receipts preserved; not used for initialization",
  "pending_gates": ["new SSL slice sampler and 1000 sample smoke", "no-label-access proof", "official-to-student feature equivalence", "single GPU smoke", "distributed smoke", "same-run resume", "topology feasibility"]
}
EOF
cat > "$run/audit/data_coverage_report.md" <<'EOF'
# Full MitoVerse coverage audit

The catalog and on-disk Zarr directory path sets matched exactly at manifest
build time. The new manifest contains 212 unique volume IDs, 212 unique paths,
13 sources, and 48,722 legal Z slices. OpenOrganelle contributes 157 volumes.
Historical split is recorded only as `legacy_split`. The manifest builder did
not use this field to decide inclusion. All catalog volumes entered the SSL
sampling population definition. This does not mean every voxel has been used
in training; Stage 1 training has not started.

The reader audit successfully read first, middle, and last full image planes
from each volume: 636 reads, zero failures. The audit did not inspect labels
or validate every underlying chunk.

Manifest SHA256: `3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9`.
EOF
cat > "$run/reports/stage1_summary.md" <<'EOF'
# Stage 1 interim status

Stage 1 is in preparation. No `B_full` optimizer step has run. The separate
2026-09-28 B/L/T full-restart service was stopped after checking process
identity; old checkpoint receipts were preserved and LT was not dispatched.

The fresh manifest covers 212 volumes from 13 sources, including 157
OpenOrganelle volumes. Catalog and disk path sets matched. Its SHA256 is
`3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9`.
Representative image slice reads passed for all 212 volumes (636 reads).

The official natural ViT-L/16 weight SHA256 matched
`8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`.
No old P1/P2 or B/L/T/LT checkpoint was used to initialize this Stage 1 run.
`B_full` initialization, smoke tests, 100k training, and topology feasibility
are pending. No representation improvement claim is made.
EOF
systemctl show mitoverse-full-seed0.service -p ActiveState -p MainPID --no-pager
cat "$run/data/manifest_summary.json"
