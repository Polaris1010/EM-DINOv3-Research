set -euo pipefail
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,numpy as np,psutil,urllib.request
r=Path('/opt/data/3_117/tanghr/codex/primus_official_reprobe_20261013')
print('STATUS',(r/'STATUS.json').read_text())
for p in sorted((r/'runs').glob('*Dataset131_CMCT.log')):
 print(p.name,'\n',p.read_text()[-1700:])
base=Path('/opt/data/3_117/tanghr/Med_DINOv3/nnUNet_preprocessed/Dataset131_CMCT')
split=json.loads((base/'splits_final.json').read_text())[0]
print('CMCT_PREPROCESSED_SHAPES',{case:np.load(base/'nnUNetPlans_2d'/(case+'.npy'),mmap_mode='r').shape for case in split['val']})
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
print('CPU',psutil.cpu_percent(interval=1),'RAM_FREE_GiB',round(psutil.virtual_memory().available/2**30,2))
with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=5) as response:print('WEBSITE',response.status)
PY
