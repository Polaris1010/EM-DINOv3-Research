"""Feasibility only: short default per-GPU batch=16 test on the adapted ViT-L."""
import subprocess,time,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python';SSL='/opt/data/3_117/tanghr/dinov3_shared/bin/python'
while not (BASE/'phase2_gpu3_complete.json').exists():time.sleep(10)
name='ssl_b16_memory_smoke';assert not (BASE/'jobs'/name).exists()
cmd=[PY,'-u',str(BASE/'guard.py'),'--gpu','3','--run',name,'--python',SSL,str(BASE/'ssl_sweep.py'),'--steps','10','--batch','16','--gram-weight','2']
code=subprocess.call(cmd,cwd=BASE);d=BASE/'jobs'/name
g=json.loads((d/'guard_result.json').read_text()) if (d/'guard_result.json').exists() else {'exit_code':code,'stop_reason':'preflight did not finish'}
r=json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else None
report={'purpose':'memory and execution feasibility only; no downstream score; adapted ViT-L heads, not literal ViT-7B recipe','guard':g,'result':r}
(BASE/'batch16_feasibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
