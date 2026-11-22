#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>run.lock
flock -n 9
card=1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > gpu_before.txt
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader >> gpu_before.txt
gpu=$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)
test "$(nvidia-smi -i "$card" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100
if grep -Fq "$gpu" gpu_before.txt; then exit 1; fi
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
P=/opt/data/3_117/tanghr/envs/meddinov3/bin/python
"$P" -u run.py
/usr/bin/python3 report.py
