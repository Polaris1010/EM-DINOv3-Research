set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
test -z "$(git status --porcelain)"
test ! -e branch_E/control_20261011
test ! -e branch_S/blind_review_raw_20261011
test "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)" = 200
mkdir -p branch_E/control_20261011 branch_S/blind_review_raw_20261011
cp STATUS.json branch_E/control_20261011/status_before.json
{
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  free -h
} > branch_E/control_20261011/preflight.txt
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('STATUS.json');d=json.loads(p.read_text())
d['P']='PAUSED_BY_USER_PENDING_MEASUREMENT_DEFINITION'
d['E']='B_FULL_COMPLETE_OFFICIAL_NATURAL_CONTROL_IN_PROGRESS'
d['S_raw_review']='PREPARING_100_RAW_PANELS_AND_20_PAIR_TRIAL'
d['long_training']='PROHIBITED'
p.write_text(json.dumps(d,indent=2))
PY
echo setup_complete
