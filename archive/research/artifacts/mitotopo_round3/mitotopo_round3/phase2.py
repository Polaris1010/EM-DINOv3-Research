"""Second bounded parameter batch; waits for the first downstream pipeline per GPU."""
import subprocess,json,time,argparse
from pathlib import Path
BASE=Path(__file__).resolve().parent
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
SSL='/opt/data/3_117/tanghr/dinov3_shared/bin/python'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[1,2,3],required=True);a=ap.parse_args()
 while not (BASE/f'pipeline_gpu{a.gpu}_complete.json').exists():time.sleep(10)
 name,batch,weight,factor={1:('ssl_b4_w0_s2000',4,0,4),2:('ssl_b2_w2_lr1_s2000',2,2,1),3:('ssl_b4_w2_s2000',4,2,4)}[a.gpu]
 jobs=[(name,SSL,'ssl_sweep.py',['--steps','2000','--batch',str(batch),'--gram-weight',str(weight),'--lr-factor',str(factor)]),('probe_'+name,PY,'probe.py',['--encoder',str(BASE/'jobs'/name/'teacher_latest.pth')])]
 for run,python,script,args in jobs:
  assert not (BASE/'jobs'/run).exists()
  cmd=[PY,'-u',str(BASE/'guard.py'),'--gpu',str(a.gpu),'--run',run,'--python',python,str(BASE/script),*args]
  print('START',run,flush=True);code=subprocess.call(cmd,cwd=BASE);g=json.loads((BASE/'jobs'/run/'guard_result.json').read_text())
  assert code==0 and g['exit_code']==0 and g['stop_reason'] is None,(run,g)
  print('COMPLETE',run,flush=True)
 (BASE/f'phase2_gpu{a.gpu}_complete.json').write_text(json.dumps({'status':'complete','jobs':[j[0] for j in jobs]}))
if __name__=='__main__':main()
