set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012
systemctl show mitoverse-B-full-extend150k-20261012.service -p ActiveState -p MainPID -p Result -p ExecMainStatus --no-pager
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json
r=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
for rel in ['STATUS.json','runs/B_full/formal_attempt1/result.json','runs/B_full/formal_attempt1/progress.json','evaluation/comparison_100k_150k.json']:
 p=r/rel
 print('\nFILE',rel,'EXISTS',p.exists(),flush=True)
 if p.exists(): print(p.read_text(),flush=True)
for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
 out=r/'evaluation'/('B_full150k_'+ds)
 for name in ['result.json','metrics.json']:
  p=out/name
  print('\nFILE',str(p.relative_to(r)),'EXISTS',p.exists(),flush=True)
  if p.exists(): print(p.read_text(),flush=True)
 log=r/'evaluation'/(ds+'.log')
 if log.exists():print('LOG_TAIL',ds,'\n'.join(log.read_text(errors='replace').splitlines()[-12:]),flush=True)
ck=r/'runs/B_full/formal_attempt1/checkpoints/step000150000'
for p in ck.glob('*'):print('CHECKPOINT',p.name,p.stat().st_size,flush=True)
PY
journalctl -u mitoverse-B-full-extend150k-20261012.service -n 25 --no-pager
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
