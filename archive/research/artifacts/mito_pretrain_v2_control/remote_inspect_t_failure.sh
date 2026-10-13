set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
cat "$run/src/STOP.json"
tail -n 90 "$run/src/jobs/T_seed0_batch16_smoke10/train.log"
cat "$run/src/jobs/T_seed0_batch16_smoke10/guard_result.json"
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
