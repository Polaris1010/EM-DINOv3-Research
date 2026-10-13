set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
test ! -e "$run/src/STOP.json"
systemctl --version | head -n 1
systemctl is-system-running || true
stat -c 'B_100_checkpoint_bytes=%s' "$run/src/jobs/B_seed0_batch16_stability100/checkpoint_step100.pth"
stat -c 'LT_100_checkpoint_bytes=%s' "$run/src/jobs/LT_seed0_batch16_stability100/checkpoint_step100.pth"
df -B1 --output=avail "$run" | tail -n 1
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,uuid,memory.free,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
