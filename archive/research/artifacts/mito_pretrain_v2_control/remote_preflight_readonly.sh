set -u
date -u
hostname
id
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
free -h
df -h /opt/data/3_117/tanghr
ps -eo pid,ppid,user,etimes,rss,args --sort=-rss | head -n 30
find /opt/data/3_117/tanghr/codex/mitoverse_round6_20260917_044606 -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort | head -n 50
for d in /opt/data/3_117/tanghr/DINOv3 /opt/data/3_117/tanghr/Med_DINOv3; do
  if [ -d "$d/.git" ]; then
    git -C "$d" rev-parse HEAD
    git -C "$d" status --short | head -n 20
  fi
done
find /opt/data/3_117/tanghr/codex -maxdepth 1 -type d -name 'mito_pretrain_v2_*' -printf '%f\n' 2>/dev/null | sort
