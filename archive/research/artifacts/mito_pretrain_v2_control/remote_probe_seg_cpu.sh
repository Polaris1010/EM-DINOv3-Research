set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /opt/data/3_117/tanghr/envs/meddinov3/bin/python "$run/src/probe_seg_cpu.py" \
  --manifest "$run/data/seg_train.json" | tee "$run/evidence/seg_cpu_probe.json"
