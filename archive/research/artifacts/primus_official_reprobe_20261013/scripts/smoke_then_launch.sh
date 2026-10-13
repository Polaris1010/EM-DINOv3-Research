set -euo pipefail
OUT=/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013
export CUDA_VISIBLE_DEVICES=3
export B_FULL_CHECKPOINT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/runs/B_full/formal_attempt1/checkpoints/step000100000/teacher_backbone.pth
export B_FULL_STEP=100000
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps
if test -e "$OUT/audit/smoke.log"; then mv "$OUT/audit/smoke.log" "$OUT/audit/smoke_bf16_hash_failure.log"; fi
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$OUT/scripts/smoke_official.py" > "$OUT/audit/smoke.log" 2>&1
cat "$OUT/audit/smoke.log"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -m py_compile "$OUT/src/frozen_probe_official.py" "$OUT/scripts/run_comparison.py"
systemd-run --unit=primus-official-reprobe-20261013.service --description='Official MedDINOv3 Primus P0 B100k B150k frozen comparison' --property=Restart=no --property=KillMode=control-group /usr/bin/env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$OUT/scripts/run_comparison.py"
systemctl show primus-official-reprobe-20261013.service -p ActiveState -p MainPID -p Result --no-pager
