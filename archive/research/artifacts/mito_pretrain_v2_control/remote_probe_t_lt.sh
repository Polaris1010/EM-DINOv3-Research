set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
for arm in T LT; do
  PYTHONPATH="$run/deps:$base/codex/mitotopo_20260908_round2/deps" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    "$base/envs/meddinov3/bin/python" "$run/src/probe_ssl_data.py" \
    --dinov3-root "$base/DINOv3/dinov3" \
    --config "$base/DINOv3/med_checkpoints_continual_v2/config.yaml" \
    --manifest "$run/data/pretrain_train.json" --arm "$arm" \
    | tee "$run/evidence/ssl_data_probe_${arm}.json"
done
