set -e
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
find "$ROOT/stage1_reference" -maxdepth 2 -type f -printf '%f\n'
grep -Eni 'global_crops_scale|local_crops_scale|seed' "$ROOT/stage1_reference/B_full_resolved_config.yaml" | head -30 || true
sed -n '145,157p' "$ROOT/stage1_reference/B_full_resolved_config.yaml"
