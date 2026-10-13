"""Close the identified missing no-Gram control for the best lower-LR recipe."""
import subprocess,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python';SSL='/opt/data/3_117/tanghr/dinov3_shared/bin/python'
name='ssl_b2_w0_lr1_s2000'
for run,python,script,args in [(name,SSL,'ssl_sweep.py',['--steps','2000','--batch','2','--gram-weight','0','--lr-factor','1']),('probe_'+name,PY,'probe.py',['--encoder',str(BASE/'jobs'/name/'teacher_latest.pth')])]:
 assert not (BASE/'jobs'/run).exists()
 cmd=[PY,'-u',str(BASE/'guard.py'),'--gpu','3','--run',run,'--python',python,str(BASE/script),*args]
 print('START',run,flush=True);code=subprocess.call(cmd,cwd=BASE);g=json.loads((BASE/'jobs'/run/'guard_result.json').read_text());assert code==0 and g['exit_code']==0 and g['stop_reason'] is None,(run,g)
 print('COMPLETE',run,flush=True)
(BASE/'lower_lr_control_complete.json').write_text(json.dumps({'status':'complete'}))
