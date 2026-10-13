set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
cd "$run/src"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /opt/data/3_117/tanghr/dinov3_shared/bin/python - <<'PY'
import train_seg,evaluate
print('training_worker_imports_complete')
PY
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import coordinator_probes,coordinator_finetune,coordinator_independent,coordinator_final_eval,watch_all_batches
print('downstream_imports_complete')
print('formal_status',watch_all_batches.status('formal'))
print('next_stage',watch_all_batches.next_stage())
PY
