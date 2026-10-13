set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
env OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('branch_E/control_20261011/comparison.json').read_text())['status']=='complete'
audit=json.loads(Path('branch_S/blind_review_raw_20261011/human_overlay20_summary.json').read_text())
assert (audit['total'],audit['judgeable'],audit['algorithm_agreement_count'])==(20,15,6)
assert json.loads(Path('branch_S/blind_review_raw_20261011/human_key_check.json').read_text())['status']=='pass'
fresh=json.loads(Path('branch_S/blind_review_raw_20261011/fresh_trial20_manifest.json').read_text())
assert fresh['pairs']==20 and not set(fresh['pair_ids'])&set(fresh['excluded'])
p=Path('STATUS.json');d=json.loads(p.read_text())
d['S_human_pilot_agreement']=0.4
d['S_long_training_admission']=False
d['S_raw_review']='100_RAW_PANELS_READY_20_FRESH_BLINDED_PAIRS_PENDING_HUMAN'
d['E_data_overlap_audit']='PENDING'
d['P']='PAUSED_BY_USER_PENDING_MEASUREMENT_DEFINITION'
d['long_training']='PROHIBITED'
p.write_text(json.dumps(d,indent=2))
PY
{
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
} > branch_E/control_20261011/postflight.txt
cat branch_E/control_20261011/postflight.txt
git add .
git -c user.name='Codex' -c user.email='codex@local' commit -m 'Complete matched natural frozen control and 20-pair human crop audit' >/dev/null
printf 'commit=%s\n' "$(git rev-parse --short HEAD)"
test -z "$(git status --porcelain)"
echo git_status_clean
