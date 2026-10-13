set -e
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u /opt/data/3_117/tanghr/codex/mv_generalist_stage1_20261013/scripts/verify_metric_cache.py
