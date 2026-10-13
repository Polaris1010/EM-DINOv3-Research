set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/score_s.py --limit 424
