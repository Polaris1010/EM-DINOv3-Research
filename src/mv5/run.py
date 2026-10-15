"""One bounded MV5 train/evaluate pipeline; never starts another experiment."""
import csv,fcntl,os,signal,subprocess,time,urllib.request
from pathlib import Path
import psutil
from common import R,M3,MODEL,rows,read,sha,atomic,cache,config
from scoring_queue import score_queue
OLD=M3.parent/'mv4_official_mitonet_20261014'
PY=str(OLD/'venv/bin/python');METRIC_PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'

def protect():
 roots=[M3,OLD,M3.parent/'mv4b_semantic_panoptic_20261014',M3.parent/'mv4c_input_provenance_20261015']
 paths=[p for root in roots for pattern in ['*.json','*.csv','*.md'] for p in root.glob(pattern)]
 arrays=[p for root in roots for p in root.rglob('*.npy') if 'cache' not in p.relative_to(root).parts]
 checkpoints=list(M3.glob('runs/*/seed*/head_step*.pth'))+[OLD/'assets/MitoNet_v1.pth']
 return {'sha256':{str(p):sha(p) for p in paths},'asset_stat':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in arrays+checkpoints}}

def website():
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10) as f:assert f.status==200

class Controller:
 def __init__(self):
  self.gpus={int(r[0]):{'uuid':r[1].strip(),'free':int(r[2])} for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
  assert self.gpus[3]['free']>20000;website();assert read(R/'smoke/smoke_complete.json')['status']=='PASS'
  self.before=protect();atomic(R/'evidence/historical_inputs_before.json',self.before)
  self.children=[];self.handles=[];self.started=time.monotonic();self.phase='training'
 def launch(self,args,name,gpu=None):
  env={**os.environ,'MV5_ROOT':str(R),'CUDA_VISIBLE_DEVICES':self.gpus[gpu]['uuid'] if gpu is not None else '',
   'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMBA_CACHE_DIR':str(R/'cache/numba'),'CUPY_CACHE_DIR':str(R/'cache/cupy'),'PYTHONDONTWRITEBYTECODE':'1'}
  f=(R/'logs'/f'{name}.log').open('a');self.handles.append(f)
  exe=METRIC_PY if Path(args[0]).name in ['score.py','report.py'] else PY
  p=subprocess.Popen([exe,'-u',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);self.children.append(p);return p
 def health(self):
  assert time.monotonic()-self.started<config()['max_runtime_hours']*3600,'Finite MV5 runtime exceeded'
  website();progress=read(R/'training/progress.json') if (R/'training/progress.json').exists() else {}
  atomic(R/'STATUS.json',{'status':'running','phase':self.phase,'train_step':progress.get('step',0),'train_progress':progress,'seconds':time.monotonic()-self.started,'physical_training_gpu':3,'physical_evaluation_gpus':[1,2,3],'gpu0_used':False,'website_http':200,'inferred_regions':len(list((R/'runs'/MODEL).glob('*_inference.json'))),'scored_regions':len(list((R/'runs'/MODEL).glob('*_metrics.json'))),'child_pids':[p.pid for p in self.children if p.poll() is None]})
 def wait(self,children):
  while True:
   self.health()
   for p in children:
    if p.poll() not in [None,0]:raise RuntimeError(f'Own MV5 worker {p.pid} failed with {p.returncode}; inspect its log')
   if all(p.poll()==0 for p in children):return
   time.sleep(5)
 def run(self):
  assert not (R/'training/training_curve.jsonl').exists(),'No automatic retraining/overwrite'
  self.wait([self.launch([R/'code/train.py'],'train_gpu3',3)])
  assert read(R/'training/training_complete.json')['completed_steps']==40000
  self.phase='adapted_official_inference';groups=[[],[],[]];load=[0,0,0]
  for row in sorted(rows(),key=lambda r:-cache(r)['valid_voxels']):
   k=min(range(3),key=lambda i:load[i]);groups[k].append(row['id']);load[k]+=cache(row)['valid_voxels']
  self.wait([self.launch([R/'code/infer.py','--regions',*ids],f'inference_gpu{gpu}',gpu) for ids,gpu in zip(groups,[1,2,3])])
  self.phase='development_metrics';tasks=[{'region_id':r['id'],'memory_bytes':45*cache(r)['valid_voxels']+(4<<30)} for r in rows()]
  score_queue(tasks,lambda t,g:self.launch([R/'code/score.py','--region',t['region_id']],f"score_{t['region_id']}",g),self.health,lambda:psutil.virtual_memory().available)
  self.phase='report';self.wait([self.launch([R/'code/report.py'],'report')])
  after=protect();assert self.before==after,'Protected historical input changed';atomic(R/'evidence/historical_inputs_after.json',after);website()
  atomic(R/'STATUS.json',{'status':'complete','training_steps':40000,'seed':config()['seed'],'inferred_regions':38,'scored_regions':38,'sources':12,'gpu0_used':False,'website_http_after':200,'historical_inputs_unchanged':True,'further_experiments_started':False,'report':'MV5_final_report.md'})
 def stop_own(self):
  for p in self.children:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  for p in self.children:
   if p.poll() is None:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  for f in self.handles:f.close()

if __name__=='__main__':
 R.mkdir(exist_ok=True);(R/'logs').mkdir(exist_ok=True);lock=(R/'RUN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ctl=None
 try:ctl=Controller();ctl.run()
 except BaseException as e:
  atomic(R/'STATUS.json',{'status':'failed','phase':ctl.phase if ctl else 'initialize','error':repr(e),'further_experiments_started':False});raise
 finally:
  if ctl:ctl.stop_own()
