set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
systemctl show mito-pretrain-v2-seed0 -p ActiveState -p SubState -p MainPID -p Result --no-pager
if [ -f "$run/src/STOP.json" ]; then cat "$run/src/STOP.json"; fi
if [ -f "$run/src/formal_seed0_status.json" ]; then cat "$run/src/formal_seed0_status.json"; fi
if [ -f "$run/src/process_registry.json" ]; then cat "$run/src/process_registry.json"; fi
for arm in B L T LT; do
  for path in "$run"/src/jobs/formal_seed0_${arm}_part*/progress.json; do
    if [ -f "$path" ]; then echo "$path"; cat "$path"; fi
  done
done
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
tail -n 12 "$run/src/formal_coordinator.log"
