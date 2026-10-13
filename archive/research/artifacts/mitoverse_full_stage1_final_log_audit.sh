set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,math
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
out=run/'runs/B_full/formal_attempt1'
result={}
for rank in range(3):
 p=out/f'steps_rank{rank}.jsonl'
 count=0; previous=0; bad_step=0; nonfinite=0; volumes=set(); losses=[]
 with p.open() as f:
  for line in f:
   x=json.loads(line)
   count+=1
   step=int(x['step'])
   if step != previous+1: bad_step+=1
   previous=step
   for key in ('loss','grad_norm','lr','teacher_momentum','teacher_temp'):
    if not math.isfinite(float(x[key])): nonfinite+=1
   for value in x.get('loss_components',{}).values():
    if not math.isfinite(float(value)): nonfinite+=1
   if x.get('data_first_volume'): volumes.add(x['data_first_volume'])
   losses.append(float(x['loss']))
 result[f'rank{rank}']={'lines':count,'last_step':previous,'step_gaps':bad_step,'nonfinite_scalars':nonfinite,'first_sample_unique_volumes':len(volumes),'loss_first':losses[0],'loss_last':losses[-1]}
 assert count==100000 and previous==100000 and bad_step==0 and nonfinite==0
print(json.dumps(result,indent=2))
(run/'audit/final_training_log_audit.json').write_text(json.dumps(result,indent=2))
PY
