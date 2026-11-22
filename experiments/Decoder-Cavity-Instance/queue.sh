#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=/opt/data/3_117/tanghr/envs/meddinov3/bin/python
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > gpu_before.txt
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > gpu_processes_before.txt
cards=()
for i in 1 2 3; do
    uuid=$(nvidia-smi -i "$i" --query-gpu=uuid --format=csv,noheader)
    if test "$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 && ! grep -Fq "$uuid" gpu_processes_before.txt; then cards+=("$uuid"); fi
done
test "${#cards[@]}" -ge 1
arms=(arm_a_multiscale_sem arm_b_multiscale_bcd)
run(){
    local phase="$1" arm="$2" dev="$3"
    if [[ "$phase" == smoke && -s "$arm/smoke_pass.json" || "$phase" == train && -s "$arm/trained.json" ]]; then return 0; fi
    CUDA_VISIBLE_DEVICES="$dev" "$py" -u train.py "$phase" "$arm" >>"$arm.$phase.log" 2>&1
}
phase(){
    if [[ "${#cards[@]}" -ge 2 ]]; then
        run "$1" "${arms[0]}" "${cards[0]}" & local a=$!
        run "$1" "${arms[1]}" "${cards[1]}" & local b=$!
        local rc=0
        wait "$a" || rc=1
        wait "$b" || rc=1
        return "$rc"
    fi
    run "$1" "${arms[0]}" "${cards[0]}"
    run "$1" "${arms[1]}" "${cards[0]}"
}
phase smoke
"$py" -c 'import json; a,b=[json.load(open(p+"/smoke_pass.json")) for p in ("arm_a_multiscale_sem","arm_b_multiscale_bcd")]; assert a["init_shared_head_sha"]==b["init_shared_head_sha"] and len(b["target_examples"])==5'
phase train
