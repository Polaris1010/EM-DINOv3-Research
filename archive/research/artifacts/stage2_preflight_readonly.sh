set -euo pipefail
stage1=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
date -Is
hostname
uptime
free -h
df -h "$stage1"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | head -n 22
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result --no-pager
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
sha256sum "$stage1/data/pretrain_all_mitoverse.json" "$stage1/runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth" | head -2
