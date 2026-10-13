set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/microsam/bin/python src/blind_panels.py
