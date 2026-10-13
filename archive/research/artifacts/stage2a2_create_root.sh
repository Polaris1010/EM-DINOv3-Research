set -euo pipefail
BASE=/opt/data/3_117/tanghr/codex
RUN="$BASE/mitoverse_full_stage2a2_20261002"
S1="$BASE/mitoverse_full_stage1_20260928_163120"
S2="$BASE/mitoverse_full_stage2_20260930_082816"
test ! -e "$RUN"
test "$(head -1 "$S2/STATUS")" = STAGE2A_GATE=FAIL
test -f "$S1/runs/B_full/formal_attempt1/checkpoints/step000100000/teacher_backbone.pth"
mkdir -p "$RUN"/{references,audit,src,branch_S/{directional_affinity,graph_cut,panels,blind_review},branch_E/{protocol,runs,results},branch_P_pending,reports}
cp "$S2/topology_calibration/sample_manifest.json" "$RUN/references/sample_manifest_424.json"
sha256sum "$RUN/references/sample_manifest_424.json" > "$RUN/references/sample_manifest_424.sha256"
cat > "$RUN/references/paths.json" <<EOF
{
  "stage1_root": "$S1",
  "stage2_root": "$S2",
  "B_full_teacher_backbone": "$S1/runs/B_full/formal_attempt1/checkpoints/step000100000/teacher_backbone.pth",
  "B_full_full_checkpoint": "$S1/runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth",
  "sam_vit_h_weight": "$S2/topology_calibration/sam_vit_h/sam_vit_h_4b8939.pth",
  "stage2_vit_h_paired_maps": "$S2/topology_calibration/sam_vit_h/paired_maps",
  "nnunet_root": "/opt/data/3_117/tanghr/Med_DINOv3",
  "previous_frozen_probe_source": "$BASE/mitotopo_20260909_round3/probe.py"
}
EOF
cat > "$RUN/STATUS.json" <<EOF
{"stage":"Stage2A2","P":"PAUSED_BY_USER_PENDING_MEASUREMENT_DEFINITION","S":"IN_PROGRESS","E":"IN_PROGRESS","long_training":"PROHIBITED","stage2_gate":"ARCHIVED_FAIL","stage1_immutable":true,"created_utc":"$(date -u +%FT%TZ)"}
EOF
cat > "$RUN/README.md" <<'EOF'
# Full-MitoVerse Stage 2A2

Stage 2A stays FAIL and read-only. Stage 1 stays read-only. Branch P is paused by the user until the augmentation measurement is defined. Branch S evaluates directional SAM affinity and localized connectivity cuts on the fixed 424 samples. Branch E establishes a frozen B_full probe using fixed datasets, decoder, seed, steps, and splits. No 100k training, full MAM cache, or GPU0 use is allowed in this stage.
EOF
cat > "$RUN/.gitignore" <<'EOF'
*.pth
*.pt
*.npz
*.npy
*.log
*.png
*.tif
*.tiff
*.pid
__pycache__/
branch_S/directional_affinity/maps/
branch_S/panels/images/
EOF
git -C "$RUN" init -q -b codex/stage2a2
printf 'root=%s\n' "$RUN"
printf 'sample_manifest_sha256=%s\n' "$(cut -d ' ' -f1 "$RUN/references/sample_manifest_424.sha256")"
printf 'stage2_readonly_status=%s\n' "$(head -1 "$S2/STATUS")"
