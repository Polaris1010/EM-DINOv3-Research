set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
cd "$run/src"
for arm in T LT; do
  if [ "$arm" = T ]; then attempt=1; else attempt=0; fi
  /opt/data/3_117/tanghr/envs/meddinov3/bin/python coordinator_smoke.py --arm "$arm" --gpu 3 --batch 16 --attempt "$attempt"
done
