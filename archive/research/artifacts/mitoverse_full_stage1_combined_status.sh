set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result --no-pager
systemctl show mitoverse-stage1-raw-ph.service -p ActiveState -p MainPID -p Result --no-pager
cat "$run/runs/B_full/formal_attempt1/progress.json"
if test -f "$run/topology_feasibility/raw_ph_progress.json"; then cat "$run/topology_feasibility/raw_ph_progress.json"; fi
wc -l "$run/topology_feasibility/metrics.csv"
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
