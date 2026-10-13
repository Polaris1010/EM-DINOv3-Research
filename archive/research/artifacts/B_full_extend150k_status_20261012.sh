set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012
systemctl show mitoverse-B-full-extend150k-20261012.service mitoverse-B-full-finalize150k-20261012.service -p Id -p ActiveState -p MainPID -p Result --no-pager
cat "$RUN/STATUS.json"
if test -f "$RUN/runs/B_full/formal_attempt1/initialization.json"; then cat "$RUN/runs/B_full/formal_attempt1/initialization.json"; fi
if test -f "$RUN/runs/B_full/formal_attempt1/progress.json"; then cat "$RUN/runs/B_full/formal_attempt1/progress.json"; fi
tail -8 "$RUN/runs/B_full/formal_attempt1_train.log" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
curl -s -o /dev/null -w 'website_http=%{http_code}\n' --max-time 8 http://127.0.0.1:7860/
