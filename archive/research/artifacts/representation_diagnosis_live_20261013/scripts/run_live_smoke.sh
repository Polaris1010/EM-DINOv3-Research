set -euo pipefail
OUT=/opt/data/3_117/tanghr/codex/representation_diagnosis_live_20261013
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$OUT/scripts/live_smoke.py" > "$OUT/smoke/run.log" 2>&1
cat "$OUT/smoke/run.log"
