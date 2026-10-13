set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
systemctl show mitoverse-stage1-B-full.service -p ActiveState -p MainPID -p Result --no-pager
cat "$run/STATUS.json"
if test -f "$run/runs/B_full/formal/progress.json"; then cat "$run/runs/B_full/formal/progress.json"; else echo progress_pending; fi
if test -f "$run/runs/B_full/formal/initialization.json"; then cat "$run/runs/B_full/formal/initialization.json"; fi
tail -n 12 "$run/runs/B_full/formal_train.log" 2>/dev/null || true
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
