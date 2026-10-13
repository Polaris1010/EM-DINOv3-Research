set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
date -Is
uptime
free -h
vmstat 1 3
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
ps -eo pid,ppid,pcpu,pmem,rss,comm,args --sort=-pcpu | head -n 18
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result --no-pager
cat "$run/runs/B_full/formal_attempt1/progress.json"
df -h "$run"
