set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
OUT=branch_E/boundary_metrics_20261012
tail -12 "$OUT/run.log"
if test -f "$OUT/results.json"; then cat "$OUT/REPORT_zh.md"; fi
if test -f "$OUT/error.txt"; then cat "$OUT/error.txt"; fi
ps -p "$(cat "$OUT/job.pid")" -o pid,etime,%cpu,%mem,rss,args || true
