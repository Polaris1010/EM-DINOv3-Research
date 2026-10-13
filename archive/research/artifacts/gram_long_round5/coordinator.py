import os,sys,json,time,traceback,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from guard import Guard
BASE=Path(__file__).resolve().parent;ROOT=Path('/opt/data/3_117/tanghr');SSL=str(ROOT/'dinov3_shared/bin/python')
def save(n,x):
 p=BASE/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,default=str));t.replace(p)
def event(message,**kw):
 with (BASE/'coordinator_events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),message=message,**kw))+'\n')
 print(message,kw,flush=True)
def chain_rows(p):
 rows={}
 if p.exists():
  for line in p.read_text().splitlines():
   try:r=json.loads(line);rows[r['step']]=r
   except json.JSONDecodeError:pass
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=int,default=16);ap.add_argument('--revision',default='v1');a=ap.parse_args();g=Guard();g.start();prefix=f'{a.revision}_b{a.batch}'
 def ssl(arm,gpu,phase,steps,resume=None):
  args=['--weight',2 if arm=='G' else 0,'--batch',a.batch,'--steps',steps]
  if resume:args+=['--resume',resume]
  return g.job(f'{prefix}_{arm}_{phase}',gpu,'ssl_train.py',args,python=SSL)
 def paircheck(x,y):
  xx=chain_rows(x/'steps.jsonl');yy=chain_rows(y/'steps.jsonl');common=sorted(xx.keys()&yy.keys());assert common
  for s in common:assert xx[s]['consumption_chain']==yy[s]['consumption_chain'],f'G/C input mismatch at {s}'
  event('Paired consumed-input hashes match',first_step=common[0],last_step=common[-1])
 try:
  save('launch_status.json',dict(status='functional_checks',batch=a.batch,prefix=prefix,GPUs=dict(G=3,C=1,X1=2)))
  with ThreadPoolExecutor(max_workers=3) as ex:
   fg=ex.submit(ssl,'G',3,'functional',11);fc=ex.submit(ssl,'C',1,'functional',11);fx=ex.submit(g.job,f'{prefix}_X1_smoke',2,'downstream.py',['--x1','--steps',100])
   G=fg.result();C=fc.result();fx.result()
  paircheck(G,C)
  for arm,path,gpu in [('G',G,3),('C',C,1)]:
   r=ssl(arm,gpu,'resume11',11,path/'checkpoint_step10.pth');g.job(f'{prefix}_{arm}_resume_compare',None,'compare_resume.py',[path/'checkpoint_step11.pth',r/'checkpoint_step11.pth'])
  event('Functional and resume checks passed')
  save('launch_status.json',dict(status='steady_500',batch=a.batch,prefix=prefix))
  with ThreadPoolExecutor(max_workers=2) as ex:
   fg=ex.submit(ssl,'G',3,'steady500',500,G/'checkpoint_step11.pth');fc=ex.submit(ssl,'C',1,'steady500',500,C/'checkpoint_step11.pth');G=fg.result();C=fc.result()
  paircheck(G,C);event('Both arms completed steady500; starting long continuation')
  save('launch_status.json',dict(status='long_training',batch=a.batch,prefix=prefix,target_steps=30000))
  def explore():
   for seed in [0,1,2]:
    p=g.job(f'{prefix}_X1_seed{seed}',2,'downstream.py',['--x1','--seed',seed]);g.job(f'{prefix}_X1_seed{seed}_eval',2,'downstream.py',['--x1','--seed',seed,'--evaluate-from',p])
  with ThreadPoolExecutor(max_workers=3) as ex:
   fg=ex.submit(ssl,'G',3,'long30000',30000,G/'checkpoint_step500.pth');fc=ex.submit(ssl,'C',1,'long30000',30000,C/'checkpoint_step500.pth');fx=ex.submit(explore)
   gp=BASE/'jobs'/f'{prefix}_G_long30000';cp=BASE/'jobs'/f'{prefix}_C_long30000';last=0
   while not fg.done() or not fc.done():
    if g.abort.wait(30):raise RuntimeError(g.reason)
    xx=chain_rows(gp/'steps.jsonl');yy=chain_rows(cp/'steps.jsonl');common=sorted(xx.keys()&yy.keys())
    for s in common:
     if s>last:
      if xx[s]['consumption_chain']!=yy[s]['consumption_chain']:g.stop_all(f'Input hash mismatch at {s}');raise RuntimeError('Paired input mismatch')
      last=s
   G=fg.result();C=fc.result();fx.result()
  paircheck(G,C)
  def probe(arm,path,gpu):
   for milestone,seeds in [(10000,[0]),(20000,[0]),(30000,[0,1,2])]:
    for seed in seeds:
     args=['--encoder',path/f'teacher_step{milestone}.pth','--seed',seed];name=f'{prefix}_{arm}_{milestone}_head{seed}'
     p=g.job(name,gpu,'downstream.py',args);g.job(name+'_eval',gpu,'downstream.py',args+['--evaluate-from',p]+(['--validation-only'] if milestone<30000 else []))
  save('launch_status.json',dict(status='downstream_probes',batch=a.batch,prefix=prefix))
  with ThreadPoolExecutor(max_workers=2) as ex:
   fs=[ex.submit(probe,'G',G,3),ex.submit(probe,'C',C,1)]
   for f in fs:f.result()
  save('launch_status.json',dict(status='complete_pending_review',batch=a.batch,prefix=prefix,X2_X3='not run'));event('Authorized round finished; stopped for review')
 except BaseException as e:
  g.stop_all(repr(e));save('launch_status.json',dict(status='stopped',reason=repr(e),batch=a.batch,prefix=prefix));traceback.print_exc();raise
 finally:g.close()
if __name__=='__main__':main()
