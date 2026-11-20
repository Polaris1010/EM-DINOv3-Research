#!/bin/bash
set -eu
cd "$(dirname "$0")"
test "$(hostname -s)" = mira-4090-8
exec 9>run.lock
flock -n 9
py=../mv4_official_mitonet_20261014/venv/bin/python
export PYTHONPATH="$(realpath ../mitonet_paper_20261119/deps):$(realpath ../mitonet_paper_20261119/empanada-main)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p scores pred
one(){
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)"
    "$py" run.py "$2" >"worker_$2.log" 2>&1
}
one 1 0 & a=$!
one 3 1 & b=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
test "$rc" = 0
"$py" run.py report
