set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
test "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)" = 200
test "$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1024
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/microsam/bin/python src/extract_directional.py --limit 1
