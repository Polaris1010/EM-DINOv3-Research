set -euo pipefail
UNIT=mv-dev-v2-validation-20261013.service
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013
PID=$(systemctl show "$UNIT" --property=MainPID --value)
if test "$PID" -gt 0; then
 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$PID" <<'PY'
import sys,psutil
p=psutil.Process(int(sys.argv[1]));assert '/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013/scripts/run_validation.py' in p.cmdline()
print('STOP_OWN_SUPERSEDED_JOB',p.pid)
PY
 systemctl stop "$UNIT"
fi
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,urllib.request
r=Path('/opt/data/3_117/tanghr/codex/mv_dev_v2_expansion_20261013')
(r/'STATUS_TTA_SUPERSEDED.json').write_text(json.dumps({'status':'superseded','reason':'User correction: MV2 must inherit MV1 TTA OFF; four-flip outputs excluded','history_preserved':True},indent=2))
print('WEBSITE',urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8).status)
PY
