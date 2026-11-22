#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec 8>eval.lock
flock -n 8
py=/opt/data/3_117/tanghr/envs/meddinov3/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
parent="$1"
one(){
    local arm="$1"
    while ! test -s "$arm/trained.json"; do
        kill -0 "$parent" 2>/dev/null || return 1
        sleep 10
    done
    local card uuid attempts=0
    card=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpu"])' "$arm/trained.json")
    case "$card" in 1|2|3) ;; *) return 13;; esac
    uuid=$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)
    exec {slot}>"gpu_${card}_eval.lock"
    flock "$slot"
    while test "$(nvidia-smi -i "$card" --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 100 || nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader|grep -Fxq "$uuid"; do
        sleep 10
        attempts=$((attempts+1))
        test "$attempts" -lt 120 || return 14
    done
    CUDA_VISIBLE_DEVICES="$uuid" "$py" -u evaluate_new.py "$arm" >>"$arm.eval.log" 2>&1
}
one arm_a_multiscale_sem & a=$!
one arm_b_multiscale_bcd & b=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
test "$rc" = 0
card=1
uuid=$(nvidia-smi -i "$card" --query-gpu=uuid --format=csv,noheader)
if test "$(nvidia-smi -i "$card" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 && ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader|grep -Fxq "$uuid"; then
    CUDA_VISIBLE_DEVICES="$uuid" "$py" -u panels.py >>panels.log 2>&1
fi
"$py" summarize.py
