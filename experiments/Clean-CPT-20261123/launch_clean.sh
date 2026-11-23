#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
project=$(cd "$root/../.." && pwd)
python="$project/dinov3_shared/bin/python"
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$project/codex/mitotopo_20260908_round2/deps:$project/DINOv3/dinov3"
mode=$1
arm=$2
host=$(hostname -s)
case "$host:$arm" in
  mira-4090-8:Clean_C0_100k|mira-4090-7:Clean_C1_0k_30k|mira-4090-7:Clean_C1_30k_to_100k) ;;
  *) echo "Unexpected host for $arm: $host" >&2; exit 1 ;;
esac
if [[ $mode == formal ]]; then
  "$python" "$root/code/check_gate.py" "$root"
  stage=runs
elif [[ $mode == smoke && $arm != Clean_C1_30k_to_100k ]]; then
  stage=smoke
else
  exit 1
fi
mkdir -p "$root/$stage" "$root/locks"
exec 9>"$root/locks/$host.lock"
flock -n 9
if [[ $arm == Clean_C1_30k_to_100k ]]; then
  for rank in 0 1 2; do
    test -s "$root/runs/Clean_C1_0k_30k/result_rank$rank.json"
  done
  until [[ -s "$root/step000030000_full_state.READY" ]]; do sleep 30; done
  "$python" "$root/code/check_gate.py" "$root" --handoff
fi
gpu_rows=$(nvidia-smi -i 1,2,3 --query-gpu=index,uuid,pci.bus_id,memory.used --format=csv,noheader,nounits)
apps=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader,nounits)
export CLEAN_GPU_UUIDS
CLEAN_GPU_UUIDS=$(printf '%s\n' "$gpu_rows" | "$python" -c '
import csv,sys
rows = [[c.strip() for c in row] for row in csv.reader(sys.stdin)]
assert [r[0] for r in rows] == ["1","2","3"], rows
assert all(int(r[3]) <= 32 for r in rows), rows
buses = {r[2].lower() for r in rows}
for line in sys.argv[1].splitlines():
    assert line.split(",")[0].strip().lower() not in buses, line
print(",".join(r[1] for r in rows))
' "$apps")
out="$root/$stage/$arm"
mkdir "$out"
printf '%s\n' "$gpu_rows" > "$out/gpu_before.csv"
exec > "$out/launcher.log" 2>&1
echo "host=$host pid=$$ arm=$arm mode=$mode CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec "$python" -m torch.distributed.run --standalone --nproc_per_node=3 "$root/code/train_clean.py" "$arm" "$mode"
