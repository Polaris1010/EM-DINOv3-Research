set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
date -u +%FT%TZ
git status --short
cat STATUS.json
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
free -h
df -h "$RUN"
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
cat references/paths.json
sha256sum src/frozen_probe.py
find /opt/data/3_117/tanghr/DINOv3/pretrained_weights -maxdepth 2 -type f -printf '%p %s bytes\n'
sed -n '1,95p' /opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/scripts/validate_initialization.py
printf 'existing_eval_status\n'
for DS in Dataset111_CMCC Dataset131_CMCT; do
  cat "branch_E/runs/B_full_${DS}/result.json"
done
