set -e
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
date -u +%FT%TZ
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
for DS in Dataset111_CMCC Dataset131_CMCT; do
  OUT="branch_E/runs/B_full_${DS}"
  printf '%s ' "$DS"
  if test -f "$OUT/result.json"; then head -c 150 "$OUT/result.json"; printf '\n'; else cat "$OUT/progress.json" 2>/dev/null || tail -2 "$OUT/train.log"; fi
done
printf 'S_score='; if test -f branch_S/graph_cut/summary_424.json; then echo complete; else tail -2 branch_S/graph_cut/score.log 2>/dev/null || true; fi
