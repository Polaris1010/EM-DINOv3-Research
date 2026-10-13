set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227/src
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import ast,json,time,hashlib
p=Path('guard.py');s=p.read_text()
old="else PYTHON),'-u',BASE/'worker.py'"
new="else '/opt/data/3_117/tanghr/envs/meddinov3/bin/python' if script=='expanded_eval.py' else PYTHON),'-u',BASE/'worker.py'"
assert s.count(old)==1 and "script=='expanded_eval.py'" not in s
patched=s.replace(old,new);ast.parse(patched)
backup=Path('../evidence')/f'guard_before_expanded_{int(time.time())}.py';backup.write_text(s)
tmp=p.with_suffix('.py.tmp');tmp.write_text(patched);tmp.replace(p)
for name in ('coordinator_probes.py','expanded_eval.py','expanded_queue.py'):ast.parse(Path(name).read_text())
coverage=json.loads(Path('../evidence/expanded_coverage.json').read_text());assert not coverage['blocked']
Path('expanded_status.json').write_text(json.dumps({'status':'queued','new_volumes':coverage['new_inference_volumes'],
 'total_volumes':coverage['eligible_volumes'],'gpu':3,'dependency':'five final-checkpoint seed0 probes complete; then run alongside remaining probes on GPUs1/2',
 'queued_at':time.time(),'legacy_interpreter':'meddinov3','model_jobs':['MitoNet','P1_seed0','P1_seed1','P1_seed2']},indent=2))
print('Expanded comparison queued; new coordinator will use existing legacy environment for old model workers.')
PY
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import torch
torch.set_num_threads(2)
net=torch.jit.load('/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/official/MitoNet_v1.pth',map_location='cpu').eval()
with torch.inference_mode():
 x=torch.randn(2,1,128,128)
 a=net(x)['sem_logits'];b=torch.cat([net(x[i:i+1])['sem_logits'] for i in range(2)])
 delta=float((a-b).abs().max());assert a.shape==(2,1,128,128) and delta<1e-4
print('MitoNet batch2 CPU compatibility passed; max logit delta',delta)
PY
