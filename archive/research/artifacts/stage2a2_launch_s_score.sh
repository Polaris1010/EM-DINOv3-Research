set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
test -f branch_S/directional_affinity/complete.json
nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/score_s.py --limit 424 > branch_S/graph_cut/score.log 2>&1 < /dev/null &
echo $! > branch_S/graph_cut/score.pid
printf 'pid=%s\n' "$(cat branch_S/graph_cut/score.pid)"
for DS in Dataset111_CMCC Dataset131_CMCT; do
  OUT="branch_E/runs/B_full_${DS}"
  printf 'eval_%s=' "$DS"
  if test -f "$OUT/progress.json"; then cat "$OUT/progress.json"; else tail -2 "$OUT/train.log"; fi
done
