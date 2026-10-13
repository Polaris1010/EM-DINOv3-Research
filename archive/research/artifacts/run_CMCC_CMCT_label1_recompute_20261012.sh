set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -m py_compile "$RUN/scripts/rescore_label1.py"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,hashlib
r=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
s=(r/'src/frozen_probe_150k.py').read_text()
assert s.count('t=np.asarray(gt.dataobj[:,:,z]).T>0')==1
s=s.replace('t=np.asarray(gt.dataobj[:,:,z]).T>0','t=np.asarray(gt.dataobj[:,:,z]).T==1')
target='b=np.array(seg[0,z,y:y+224,x:x+224],dtype=np.int64,copy=True)'
assert s.count(target)==1
s=s.replace(target,target+'\n        b=np.where(b<0,-1,(b==1).astype(np.int64))')
condition='(k==1 or isinstance(k,tuple) and 1 in k and 0 not in k)'
assert s.count(condition)==2
s=s.replace(condition,'(k==1 or k==(1,))')
compile(s,'frozen_probe_label1.py','exec')
p=r/'src/frozen_probe_label1.py';assert not p.exists();p.write_text(s)
(r/'audit/label1_frozen_probe_policy.json').write_text(json.dumps({'target':'mitochondria only','raw_GT_rule':'GT == 1','decoder_training_target':'label 1 -> 1; other nonnegative semantic labels -> 0; negative ignore labels -> -1','foreground_sampling':'only class 1 coordinates','source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'previous_preprocessed_label_classes':[-1,0,1],'old_training_effect':'no difference for the audited binary inputs'},indent=2))
PY
systemd-run --unit=mitoverse-class1-metrics-20261012.service --description='Recompute natural 100k 150k mitochondrial-only frozen metrics' --property=Restart=no --property=KillMode=control-group /usr/bin/env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u "$RUN/scripts/rescore_label1.py"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import json,subprocess,time
r=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012/evaluation/label1_only_recompute_20261012')
unit='mitoverse-class1-metrics-20261012.service';last=''
for i in range(30):
 if (r/'results.json').exists():print('LABEL1_RESULTS_COMPLETE',(r/'results.json').read_text(),flush=True);break
 if (r/'progress.json').exists():
  s=(r/'progress.json').read_text()
  if s!=last:print('PROGRESS',s,flush=True);last=s
 state=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','Result','--no-pager'],capture_output=True,text=True).stdout
 print(state.strip(),flush=True)
 if 'ActiveState=failed' in state:
  print(subprocess.run(['journalctl','-u',unit,'-n','35','--no-pager'],capture_output=True,text=True).stdout,flush=True);raise SystemExit(2)
 time.sleep(30)
else:print('WATCH_TIMEOUT',flush=True)
PY
