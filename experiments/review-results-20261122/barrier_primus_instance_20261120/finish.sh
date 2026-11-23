#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>finish.lock
flock -n 9
py=/opt/data/3_117/tanghr/envs/meddinov3/bin/python
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader)"
if [[ -n "${1:-}" ]]; then
    while [[ ! -s trained.json ]] && kill -0 "$1" 2>/dev/null; do sleep 15; done
    test -s trained.json
fi
if [[ ! -s trained_40000.json ]]; then
    "$py" -u train.py >>train_40000.log 2>&1
fi
"$py" -u evaluate_head.py >>evaluation.log 2>&1
