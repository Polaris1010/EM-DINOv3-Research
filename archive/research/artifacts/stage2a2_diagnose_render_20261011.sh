set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
sed -n '/^def read_patch/,/^def /p' /opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/scripts/mam_sam_vit_b_smoke.py
cat branch_S/blind_review_raw_20261011/render.log
find branch_S/blind_review_raw_20261011/panels -maxdepth 1 -type f -name 'P*.png' | wc -l
for DS in Dataset111_CMCC Dataset131_CMCT; do
  OUT="branch_E/control_20261011/runs/official_natural_${DS}"
  cat "$OUT/progress.json" 2>/dev/null || tail -3 "$OUT/train.log"
done
