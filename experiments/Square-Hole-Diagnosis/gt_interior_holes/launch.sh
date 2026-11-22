#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>run.lock
flock -n 9
card=1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > gpu_before.txt
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader >> gpu_before.txt
uuid=$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)
test "$(nvidia-smi -i "$card" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100
if grep -Fq "$uuid" gpu_before.txt; then exit 1; fi
export CUDA_VISIBLE_DEVICES="$uuid" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR="$PWD/mpl_cache" XDG_CACHE_HOME="$PWD/.cache"
P=/opt/data/3_117/tanghr/codex/mv4_official_mitonet_20261014/venv/bin/python
"$P" -u run.py
"$P" report.py
