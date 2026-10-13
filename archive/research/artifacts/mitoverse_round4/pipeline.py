import json,sys,time,concurrent.futures,traceback,hashlib,csv,signal
from pathlib import Path
from guard import Guard,gpu_state,apps,website
BASE=Path(__file__).resolve().parent

def save(name,x):
 p=BASE/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def release_check():
 deadline=time.time()+120
 while True:
  g=gpu_state();a=apps(g)
  if all(g[i]['used']<600 for i in [1,2,3]) and all(website(p) for i,p,m in a):return g
  if time.time()>deadline:raise RuntimeError('CUDA memory/process did not release after smoke')
  time.sleep(5)

def main():
 assert (BASE/'prepare_complete.json').exists();assert not (BASE/'STOP.json').exists(),'Inspect and fix previous stop before explicit resume'
 guard=Guard();guard.start()
 def on_signal(sig,frame):guard.stop_all(f'Coordinator signal {sig}');raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
 def parallel(stage,seed=0,norm='N1'):
  def task(pair):
   pid,gpu=pair;name=f'{stage}_{pid}_seed{seed}_{norm}'
   return guard.job(name,gpu,'train.py',['--id',pid,'--seed',seed,'--norm',norm,'--mode',stage])
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:return list(pool.map(task,[('P0',1),('P1',2),('P2',3)]))
 try:
  save('status.json',{'stage':'schedule_preparation','time':time.time()})
  guard.job('schedule_preparation',None,'data.py',[])
  save('status.json',{'stage':'smoke','time':time.time()})
  outs=parallel('smoke');res=[json.loads((o/'result.json').read_text()) for o in outs]
  for key in ['head_init_sha256','schedule_sha256','consumed_indices_sha256']:assert len({r[key] for r in res})==1
  snapshot=release_check();save('smoke_all_pass.json',{'results':res,'post_exit_gpu':snapshot})
  pilots={}
  for norm in ['N1','N2']:
   save('status.json',{'stage':'normalization_pilot','norm':norm,'time':time.time()})
   outs=parallel('pilot',norm=norm);pilots[norm]={o.name:json.loads((o/'curve.json').read_text())[-1]['source_macro_dice'] for o in outs}
  means={n:sum(v.values())/len(v) for n,v in pilots.items()};norm=max(['N1','N2'],key=lambda n:means[n]);save('normalization_selection.json',{'selected':norm,'means':means,'pilot_scores':pilots,'test_used':False,'rule':'equal average over P0 P1 P2 and full-XY validation source means; 8 evenly-spaced validation slices per volume; same500-step budget; noTTA threshold0.5 fixed a priori'})
  experiments=[]
  for s in range(3):
   save('status.json',{'stage':'formal_training','seed':s,'norm':norm,'time':time.time()})
   outs=parallel('train',s,norm)
   for o in outs:
    r=json.loads((o/'result.json').read_text());experiments.append(dict(id=r['id'],seed=s,train_dir=str(o),norm=norm,status='trained',seconds=r['seconds'],best_validation_dice=r['best_validation_dice']))
   # All formal model/normalization selection remains validation-only. Run all training seeds before any test inference.
  with (BASE/'experiments.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(experiments[0]));w.writeheader();w.writerows(experiments)
  save('evaluation_lock.json',{'normalization':norm,'threshold':.5,'tta':False,'tile':256,'step':.5,'checkpoints':[dict(id=r['id'],seed=r['seed'],path=r['train_dir']+'/head_best.pth') for r in experiments],'time':time.time(),'test_has_not_been_evaluated':True})
  for s in range(3):
   save('status.json',{'stage':'full_volume_evaluation','seed':s,'time':time.time()})
   def ev(pair):
    pid,gpu=pair;src=BASE/'jobs'/f'train_{pid}_seed{s}_{norm}'
    return guard.job(f'evaluate_{pid}_seed{s}_{norm}',gpu,'train.py',['--id',pid,'--seed',s,'--norm',norm,'--mode','evaluate','--head-from',src])
   with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(ev,[('P0',1),('P1',2),('P2',3)]))
  guard.job('final_report',None,'report.py',[])
  save('status.json',{'stage':'complete','time':time.time(),'paused_after_P0_P2':True})
 except BaseException as e:
  guard.stop_all(repr(e));save('status.json',{'stage':'stopped','reason':repr(e),'time':time.time()});raise
 finally:guard.close()
if __name__=='__main__':main()
