set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result --no-pager
cat "$run/runs/B_full/formal_attempt1/progress.json"
if test -d "$run/runs/B_full/formal_attempt1/checkpoints"; then find "$run/runs/B_full/formal_attempt1/checkpoints" -maxdepth 2 -type f -printf '%P %s bytes\n' | head -20; fi
if test -f "$run/runs/B_full/formal_attempt1/latest_checkpoint.json"; then cat "$run/runs/B_full/formal_attempt1/latest_checkpoint.json"; fi
tail -n 3 "$run/audit/gpu_guard.jsonl"
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
df -h "$run"
