set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
PYTHONPATH="$run/deps" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$base/envs/meddinov3/bin/python" -u "$run/src/calibrate_topology.py" \
  --train-manifest "$run/data/pretrain_train.json" \
  --output-dir "$run/calibration_smoke_p2" \
  --per-source 2 --seed 260926
