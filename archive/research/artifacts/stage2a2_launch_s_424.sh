set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
test "$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1024
nohup env CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/microsam/bin/python src/extract_directional.py --limit 424 > branch_S/directional_affinity/generate.log 2>&1 < /dev/null &
echo $! > branch_S/directional_affinity/generate.pid
printf 'pid=%s\n' "$(cat branch_S/directional_affinity/generate.pid)"
