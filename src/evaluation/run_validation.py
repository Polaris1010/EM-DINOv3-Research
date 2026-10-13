"""Run the nine EXISTING heads on GPUs1/2/3; CPU cache/scoring stream independently."""
from pathlib import Path
import sys,os,json,time,subprocess,csv,signal,urllib.request,traceback,tarfile
import psutil
R=Path(__file__).resolve().parents[1];OLD=R.parent/'mv_generalist_stage1_20261013'
sys.path.insert(0,str(OLD/'src'))
from mv_data import sha,atomic
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None
def website():
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8) as f:assert f.status==200
def apps():
 return list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()))
def main():
 if '--worker' in sys.argv:
  model=sys.argv[sys.argv.index('--worker')+1]
  for s in [20261013,20261014,20261015]:
   subprocess.run([PY,'-u',str(R/'src/validation.py'),'infer','--model',model,'--seed',str(s)],check=True)
  return
 website();children=[];handles=[]
 gpu={int(r[0]):{'uuid':r[1].strip(),'free':int(r[2])} for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
 assert all(gpu[i]['free']>14000 for i in [1,2,3])
 existing={int(r[1]):psutil.Process(int(r[1])).create_time() for r in apps() if r[0].strip() in [gpu[i]['uuid'] for i in [1,2,3]]}
 subprocess.run([PY,'-u',str(R/'src/validation.py'),'prepare'],check=True)
 protected=read(R/'evidence/input_hashes.json')
 def launch(args,name,uuid=''):
  f=(R/'logs'/f'{name}.log').open('a');handles.append(f)
  env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
  p=subprocess.Popen([PY,'-u',*args],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(p);return p
 try:
  cache=launch([str(R/'src/validation.py'),'cache'],'cache')
  scorer=launch([str(R/'src/validation.py'),'score'],'score')
  workers={model:launch([__file__,'--worker',model],model,gpu[i]['uuid']) for i,model in zip([1,2,3],['P0','B100k','B150k'])}
  inferred=False;t0=time.time()
  while True:
   website()
   for p in children:
    if p.poll() not in (None,0):raise RuntimeError(f'Own worker {p.pid} failed with {p.returncode}')
   for r in apps():
    if r[0].strip() not in [gpu[i]['uuid'] for i in [1,2,3]]:continue
    proc=psutil.Process(int(r[1]));known=proc.pid in existing and existing[proc.pid]==proc.create_time()
    own=any(proc.pid==c.pid or any(a.pid==c.pid for a in proc.parents()) for c in children)
    if not (known or own):raise RuntimeError('New unrelated GPU process; stop only own workers')
   if not inferred and all(p.poll()==0 for p in workers.values()):
    inferred=True;atomic(R/'inference_all_complete.json',{'status':'complete','groups':9,'seconds':time.time()-t0})
   atomic(R/'STATUS.json',{'status':'running','inference_complete':inferred,'seconds':time.time()-t0,
    'physical_gpus':[1,2,3],'workers':{k:{'pid':p.pid,'exit_code':p.poll(),'progress':[read(v) for v in sorted((R/'runs'/k).glob('seed*/progress.json'))]} for k,p in workers.items()},
    'cache':read(R/'cache_progress.json'),'scoring':read(R/'scoring_progress.json')})
   if inferred and cache.poll()==0 and scorer.poll()==0:break
   time.sleep(10)
  after={p:sha(p) for p in protected};assert after==protected;website()
  atomic(R/'evidence/preserved_history.json',{'status':'PASS','before':protected,'after':after})
  atomic(R/'STATUS.json',{'status':'complete','inference_groups':9,'metrics':read(R/'metrics_all_complete.json'),'physical_gpus':[1,2,3],'website_http_after':200})
  print('FULL_VALIDATION_COMPLETE',flush=True)
 except Exception:
  atomic(R/'STATUS.json',{'status':'failed','error':traceback.format_exc(),'own_pids':[p.pid for p in children]})
  for p in children:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  raise
 finally:
  for f in handles:f.close()
if __name__=='__main__':main()
