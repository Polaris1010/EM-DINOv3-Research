set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
date -u +%FT%TZ
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
for DS in Dataset111_CMCC Dataset131_CMCT; do
  OUT="branch_E/control_20261011/runs/official_natural_${DS}"
  echo "$DS"
  if test -f "$OUT/result.json"; then cat "$OUT/result.json"; echo; else cat "$OUT/progress.json" 2>/dev/null || true; echo; tail -2 "$OUT/train.log"; fi
done
echo raw_trial
if test -f branch_S/blind_review_raw_20261011/manifest.json; then head -12 branch_S/blind_review_raw_20261011/manifest.json; else tail -5 branch_S/blind_review_raw_20261011/render.log; fi
