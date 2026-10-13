set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
mkdir -p branch_E/boundary_metrics_20261012
if test -f branch_E/boundary_metrics_20261012/error.txt; then mv branch_E/boundary_metrics_20261012/error.txt branch_E/boundary_metrics_20261012/network_attempt_error.txt; fi
nohup env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u src/boundary_metrics_20261012.py > branch_E/boundary_metrics_20261012/run.log 2>&1 < /dev/null &
echo "$!" > branch_E/boundary_metrics_20261012/job.pid
cat branch_E/boundary_metrics_20261012/job.pid
