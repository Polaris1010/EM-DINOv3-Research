set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
test "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1024
test "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)" = 200
nohup env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/microsam/bin/python src/raw_trial_panels.py > branch_S/blind_review_raw_20261011/render.log 2>&1 < /dev/null &
echo "$!" > branch_S/blind_review_raw_20261011/render.pid
env CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/validate_control.py
env RUN_OUTPUT="$RUN/branch_E/control_20261011/official_CMCC_smoke" CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/frozen_probe_control.py --dataset Dataset111_CMCC --encoder-kind official_natural --steps 2 --seed 20260907 --smoke
cat branch_E/control_20261011/official_CMCC_smoke/result.json
