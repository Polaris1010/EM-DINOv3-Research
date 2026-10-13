set -euo pipefail
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,time,subprocess
r=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
unit='mitoverse-B-full-finalize150k-20261012.service'
for i in range(30):
 p=r/'evaluation/comparison_100k_150k.json'
 if p.exists():
  print('COMPARISON_COMPLETE',p.read_text(),flush=True)
  print('STATUS_FINAL',(r/'STATUS.json').read_text(),flush=True)
  break
 state=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','Result','--no-pager'],capture_output=True,text=True).stdout
 print('FINALIZER',state.strip(),flush=True)
 journal=subprocess.run(['journalctl','-u',unit,'-n','8','--no-pager'],capture_output=True,text=True).stdout
 print(journal,flush=True)
 if 'ActiveState=failed' in state:raise SystemExit(2)
 time.sleep(30)
else:print('FINALIZATION_WATCH_TIMEOUT',flush=True)
PY
