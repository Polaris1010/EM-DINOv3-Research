set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
test "$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 5 http://127.0.0.1:7860/)" = 200
PYTHONPATH="$run/deps" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$base/envs/meddinov3/bin/python" -u "$run/src/calibrate_topology.py" \
  --train-manifest "$run/data/pretrain_train.json" \
  --output-dir "$run/topology_calibration" \
  --per-source 32 --seed 260926
test "$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 5 http://127.0.0.1:7860/)" = 200
