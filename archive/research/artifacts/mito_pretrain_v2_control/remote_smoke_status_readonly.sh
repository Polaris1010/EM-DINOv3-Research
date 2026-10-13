set -u
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
date -u
nvidia-smi --query-gpu=index,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
tail -n 25 "$run/src/jobs/B_seed0_batch16_smoke10/train.log" 2>/dev/null || true
cat "$run/src/jobs/B_seed0_batch16_smoke10/progress.json" 2>/dev/null || true
cat "$run/src/smoke_status.json" 2>/dev/null || true
curl --silent --output /dev/null --write-out 'site_http=%{http_code}\n' --max-time 5 http://127.0.0.1:7860/
