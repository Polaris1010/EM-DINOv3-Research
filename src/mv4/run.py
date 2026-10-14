"""Finite official-MitoNet execution on the locked38 MV3 development regions."""
import csv,fcntl,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
import psutil
from common import R,M3,MODEL,rows,read,sha,atomic,cache,config
from scoring_queue import score_queue
PY=str(R/'venv/bin/python')

def gpu_info():
 return {int(r[0]):{'uuid':r[1].strip(),'free_mib':int(r[2])} for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
def website():
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10) as f:assert f.status==200
def protect():
 paths=[M3/'MV-Generalist.json',M3/'MV3_protocol.json',M3/'STATUS.json',M3/'MV3_per_region_metrics.csv',M3/'MV3_final_report.md']
 paths+=list(M3.glob('runs/*/seed*/*_metrics.json'))
 return {'sha256':{str(p):sha(p) for p in paths},'asset_stat':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for pattern in ['runs/*/seed*/head_step*.pth','runs/*/seed*/*_prediction.npy'] for p in M3.glob(pattern)}}

class Controller:
 def __init__(self):
  self.gpus=gpu_info();website();self.children=[];self.handles=[];self.start=time.monotonic();self.phase='smoke'
  assert read(M3/'STATUS.json')['status']=='complete'
  assert sha(M3/'MV3_protocol.json')==config()['mv3_protocol_sha256']
  assert all(self.gpus[i]['free_mib']>18000 for i in [1,2,3]),'Not enough free GPU memory for the locked tiling configuration'
  self.before=protect();atomic(R/'evidence/mv3_inputs_before.json',self.before)
 def launch(self,args,name,gpu=None):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':self.gpus[gpu]['uuid'] if gpu is not None else '',
   'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MV4_ROOT':str(R),
   'CUPY_CACHE_DIR':str(R/'cache/cupy'),'NUMBA_CACHE_DIR':str(R/'cache/numba'),'PYTHONDONTWRITEBYTECODE':'1'}
  f=(R/'logs'/f'{name}.log').open('a');self.handles.append(f)
  p=subprocess.Popen([PY,'-u',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);self.children.append(p);return p
 def health(self):
  assert time.monotonic()-self.start<config()['max_runtime_hours']*3600,'MV4 finite runtime limit reached'
  website();out=R/'runs'/MODEL
  atomic(R/'STATUS.json',{'status':'running','phase':self.phase,'seconds':time.monotonic()-self.start,
   'inferred_regions':len(list(out.glob('*_inference.json'))),'scored_regions':len(list(out.glob('*_metrics.json'))),
   'expected_regions':38,'sources':12,'models':['G-P0','G-B100',MODEL],'dinov3_reused':True,
   'physical_gpus':[1,2,3],'gpu0_used':False,'website_http':200,'child_pids':[p.pid for p in self.children if p.poll() is None]})
 def wait(self,children):
  while True:
   self.health()
   for p in children:
    if p.poll() not in [None,0]:raise RuntimeError(f'Own MV4 worker {p.pid} failed with {p.returncode}; inspect logs')
   if all(p.poll()==0 for p in children):return
   time.sleep(3)
 def run(self):
  rr=rows();out=R/'runs'/MODEL
  # A single full native plane from three size regimes, never chosen by GT/Dice.
  smoke=['haberl18_vol1','han24_pos2','wei20_mitoEM-R__dev']
  checks=[self.launch([R/'src/infer.py','--smoke','--regions',r],f'smoke_{r}',g) for r,g in zip(smoke,[1,2,3]) if not (R/'smoke'/(r+'_inference.json')).exists()]
  self.wait(checks);atomic(R/'evidence/smoke_complete.json',{'status':'PASS','regions':smoke,'input_only_selection':True})
  self.phase='official_mitonet_inference';groups=[[],[],[]];load=[0,0,0]
  for row in sorted(rr,key=lambda r:-cache(r)['valid_voxels']):
   if (out/(row['id']+'_inference.json')).exists():continue
   k=min(range(3),key=lambda i:load[i]);groups[k].append(row['id']);load[k]+=cache(row)['valid_voxels']
  workers=[self.launch([R/'src/infer.py','--regions',*ids],f'inference_gpu{g}',g) for ids,g in zip(groups,[1,2,3]) if ids];self.wait(workers)
  assert len(list(out.glob('*_inference.json')))==38
  self.phase='development_metrics';tasks=[{'region_id':row['id'],'memory_bytes':45*cache(row)['valid_voxels']+(4<<30)} for row in rr if not (out/(row['id']+'_metrics.json')).exists()]
  tasks.sort(key=lambda t:t['memory_bytes'])
  score_queue(tasks,lambda t,g:self.launch([R/'src/score.py','--region',t['region_id']],f"metric_{t['region_id']}",g),self.health,lambda:psutil.virtual_memory().available)
  assert len(list(out.glob('*_metrics.json')))==38
  self.phase='report';self.wait([self.launch([R/'src/report.py'],'report')])
  after=protect();assert after==self.before,'Protected MV3 input changed'
  atomic(R/'evidence/mv3_inputs_after.json',after);website()
  atomic(R/'STATUS.json',{'status':'complete','inferred_regions':38,'scored_regions':38,'sources':12,
   'dinov3_metric_rows_reused':228,'mitonet_metric_rows':38,'instance_outputs_retained':True,
   'physical_gpus':[1,2,3],'gpu0_used':False,'website_http_after':200,'mv3_inputs_unchanged':True,
   'report':'MV4_final_report.md','further_experiments_started':False})
 def stop_own(self):
  for p in self.children:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  for p in self.children:
   if p.poll() is None:
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  for f in self.handles:f.close()

if __name__=='__main__':
 R.mkdir(exist_ok=True);lock=(R/'RUN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 controller=None
 try:
  controller=Controller();controller.run()
 except BaseException as e:
  atomic(R/'STATUS.json',{'status':'failed','phase':controller.phase if controller else 'initialize','error':repr(e),'further_experiments_started':False});raise
 finally:
  if controller:controller.stop_own()
