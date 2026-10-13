set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u src/gt_crop_diagnostic_20261012.py
