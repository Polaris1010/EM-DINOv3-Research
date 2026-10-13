set -u
date -u
hostname
uptime
df -h /opt/data/3_117/tanghr
free -h
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
for service in mitoverse-full-seed0.service mito-pretrain-v2-seed0.service mito-pretrain-v2-watcher.service; do
  systemctl show "$service" -p ActiveState -p MainPID -p Result --no-pager
done
ps -eo pid,ppid,user,etimes,args | grep -E 'coordinator_formal_seed0|train_ssl.py|torchrun|dinov3|mito' | grep -v grep | head -80
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
run=/opt/data/3_117/tanghr/codex/mitoverse_full_restart_20260928_133515
if test -f "$run/src/formal_seed0_status.json"; then cat "$run/src/formal_seed0_status.json"; fi
if test -f "$run/src/STOP.json"; then cat "$run/src/STOP.json"; fi
find "$run/src/jobs" -maxdepth 2 -name progress.json -print 2>/dev/null | sort | while read -r f; do
  printf '%s ' "$f"
  /opt/data/3_117/tanghr/envs/meddinov3/bin/python -c 'import json,sys; x=json.load(open(sys.argv[1])); print({k:x.get(k) for k in ("step","loss","unique_volumes")})' "$f"
done
