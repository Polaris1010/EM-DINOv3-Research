set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
for ATTEMPT in $(seq 1 120); do
  COMPLETE=1
  for DS in Dataset111_CMCC Dataset131_CMCT; do
    OUT="branch_E/control_20261011/runs/official_natural_${DS}"
    if ! test -f "$OUT/result.json"; then
      COMPLETE=0
      if ! kill -0 "$(cat "$OUT/train.pid")" 2>/dev/null; then
        tail -15 "$OUT/train.log"
        exit 31
      fi
    fi
  done
  if test "$COMPLETE" = 1; then break; fi
  sleep 5
done
test "$COMPLETE" = 1
env OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/compare_natural_control.py
