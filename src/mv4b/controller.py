"""Finite four-region pilot, then only explicitly recorded full extension."""
import argparse,csv,fcntl,os,signal,subprocess,time,urllib.request
from pathlib import Path
from shared import R,M4,M3,read,sha,atomic,rows,cache,config,output
from scoring_queue import score_queue
PY=str(M4/'venv/bin/python');CPU='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
def website():
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10) as f:assert f.status==200
def protect():
 paths=[M3/'MV-Generalist.json',M3/'MV3_protocol.json',M3/'STATUS.json',M3/'MV3_per_region_metrics.csv',M4/'MV4_protocol.json',M4/'STATUS.json',M4/'MV4_final_report.md',M4/'MV4_per_region_metrics.csv',M4/'MV4_summary.json']
 paths+=list(M4.glob('runs/*/*_metrics.json'))+list(M4.glob('runs/*/*_inference.json'))
 arrays=list(M4.glob('runs/*/*.npy'))+list(M3.glob('runs/*/seed*/*_prediction.npy'))
 return {'sha256':{str(p):sha(p) for p in paths},'array_stat':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in arrays}}
class Controller:
 def __init__(self,mode):
  self.mode=mode;self.start=time.monotonic();self.children=[];self.handles=[];self.phase='initialize';self.c=config()
  self.gpus={int(r[0]):{'uuid':r[1].strip(),'free_mib':int(r[2])} for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
  assert all(self.gpus[g]['free_mib']>18000 for g in [1,2,3]);website()
  assert read(M4/'STATUS.json')['status']=='complete';self.before=protect();atomic(R/f'evidence/{mode}_inputs_before.json',self.before)
 def launch(self,args,name,gpu=None):
  env={**os.environ,'MV4_ROOT':str(M4),'MV4B_ROOT':str(R),'CUDA_VISIBLE_DEVICES':self.gpus[gpu]['uuid'] if gpu else '',
   'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
   'CUPY_CACHE_DIR':str(R/'cache/cupy'),'NUMBA_CACHE_DIR':str(R/'cache/numba')}
  f=(R/'logs'/f'{name}.log').open('a');self.handles.append(f)
  executable=PY if Path(args[0]).name=='infer.py' else CPU
  p=subprocess.Popen([executable,'-u',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);self.children.append(p);return p
 def health(self):
  assert time.monotonic()-self.start<7200;website()
  atomic(R/'STATUS.json',{'status':'running','mode':self.mode,'phase':self.phase,'inferred_regions':len(list(output().glob('*_inference.json'))),'scored_regions':len(list(output().glob('*_metrics.json'))),'pilot_path_comparisons':len(list(output().glob('*_path_comparison.json'))),'physical_gpus':[1,2,3],'gpu0_used':False,'website_http':200,'child_pids':[p.pid for p in self.children if p.poll() is None]})
 def wait(self,ps):
  while True:
   self.health()
   if any(p.poll() not in [None,0] for p in ps):raise RuntimeError('Own MV4B worker failed; inspect logs')
   if all(p.poll()==0 for p in ps):return
   time.sleep(3)
 def run(self):
  ids=self.c['pilot_regions'] if self.mode=='pilot' else [r['id'] for r in rows()]
  if self.mode=='full':assert read(R/'extension_decision.json')['extend_to_38'] is True
  rr=[r for r in rows() if r['id'] in ids];groups=[[],[],[]];load=[0,0,0];self.phase='semantic_inference'
  for row in sorted(rr,key=lambda r:-cache(r)['valid_voxels']):
   if (output()/(row['id']+'_inference.json')).exists():continue
   k=min(range(3),key=lambda j:load[j]);groups[k].append(row['id']);load[k]+=cache(row)['valid_voxels']
  self.wait([self.launch([R/'src/infer.py','--regions',*regions],f'{self.mode}_inference_gpu{g}',g) for regions,g in zip(groups,[1,2,3]) if regions])
  self.phase='path_comparison' if self.mode=='pilot' else 'full_development_metrics'
  if self.mode=='pilot':
   self.wait([self.launch([R/'src/score.py','--region',r['id']],f'pilot_score_{r["id"]}') for r in rr])
   pilot=[read(output()/(r['id']+'_path_comparison.json')) for r in rr]
   atomic(R/'MV4B_pilot_summary.json',{'status':'complete','regions':pilot,'expansion':'Await one recorded decision on output-path differences; no GT threshold tuning.'})
  else:
   import psutil
   tasks=[{'region_id':r['id'],'memory_bytes':45*cache(r)['valid_voxels']+(4<<30)} for r in rr if not (output()/(r['id']+'_metrics.json')).exists()]
   tasks.sort(key=lambda t:t['memory_bytes'])
   score_queue(tasks,lambda t,g:self.launch([R/'src/score.py','--region',t['region_id'],'--distances'],f'metric_{t["region_id"]}',g),self.health,lambda:psutil.virtual_memory().available)
   self.phase='report';self.wait([self.launch([R/'src/report.py'],'full_report')])
  after=protect();assert after==self.before;atomic(R/f'evidence/{self.mode}_inputs_after.json',after);website()
  atomic(R/'STATUS.json',{'status':'pilot_complete' if self.mode=='pilot' else 'complete','mode':self.mode,'expected_regions':len(ids),'inferred_regions':len(list(output().glob('*_inference.json'))),'scored_regions':len(list(output().glob('*_metrics.json'))),'pilot_path_comparisons':len(list(output().glob('*_path_comparison.json'))),'historical_mv3_mv4_unchanged':True,'gpu0_used':False,'website_http_after':200,'further_experiments_started':False})
 def stop(self):
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
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','full'],required=True);a=p.parse_args();R.mkdir(exist_ok=True,parents=True);(R/'logs').mkdir(exist_ok=True);lock=(R/'RUN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 controller=None
 try:controller=Controller(a.mode);controller.run()
 except BaseException as e:atomic(R/'STATUS.json',{'status':'failed','mode':a.mode,'error':repr(e)});raise
 finally:
  if controller:controller.stop()
