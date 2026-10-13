set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
for DS in Dataset111_CMCC Dataset131_CMCT; do
  OUT="branch_E/runs/B_full_${DS}"
  echo "$DS"
  cat "$OUT/result.json" 2>/dev/null || cat "$OUT/progress.json" 2>/dev/null || true
  echo
  if test -f "$OUT/metrics.json"; then cat "$OUT/metrics.json"; fi
  echo
done
