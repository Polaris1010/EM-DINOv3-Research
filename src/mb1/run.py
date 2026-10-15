"""Finite eight-run inference/score/report controller, only physical GPUs1/2/3."""
import csv,fcntl,os,signal,subprocess,time,urllib.request
from pathlib import Path
from common import R,M3,M4,M5,OLD,read,sha,atomic,config
DINO_PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python';MITO_PY=str(M4/'venv/bin/python')
def website():
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10) as f:assert f.status==200
def protect():
 files=[p for root in [M3,M4,M5] for glob in ['*.json','*.csv','*.md'] for p in root.glob(glob)]
 files += list((OLD/'src').glob('*.py'))+list((M4/'assets').glob('*.py'))+list((M4/'assets').glob('*.yaml'))
 assets=[Path(p) for p in read(R/'evidence/checkpoint_verification.json')['assets']]
 assets += [Path(f['existing_path']) for f in read(R/'evidence/original_file_index.json')['files']]
 assets += [p for root in [M3,M4,M5] for p in root.glob('runs/**/*_prediction.npy')]
 return {'source_report_metric_sha256':{str(p):sha(p) for p in files},'asset_stat':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in assets}}
class Controller:
 def __init__(self):
  self.c=config();assert read(R/'prepare_complete.json')['status']=='PASS';website()
  for rid in ['P0','B0','O','M']:assert read(R/'smoke'/rid/'inference_complete.json')['status']=='PASS'
  self.gpus={int(x[0]):{'uuid':x[1].strip(),'free':int(x[2])} for x in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
  assert all(self.gpus[g]['free']>18000 for g in [1,2,3]);self.children=[];self.handles=[];self.start=time.monotonic();self.phase='inference_and_scoring';self.before=protect();atomic(R/'evidence/historical_inputs_before.json',self.before)
 def launch(self,run_id,mode,gpu=None):
  spec=next(s for s in self.c['runs'] if s['run_id']==run_id) if run_id else None
  env={**os.environ,'MB1_ROOT':str(R),'CUDA_VISIBLE_DEVICES':self.gpus[gpu]['uuid'] if gpu is not None else '', 'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','NUMBA_CACHE_DIR':str(R/'cache/numba'),'CUPY_CACHE_DIR':str(R/'cache/cupy')}
  exe=MITO_PY if mode=='infer' and spec['family']=='mitonet' else DINO_PY
  f=(R/'logs'/f'{mode}_{run_id or "final"}.log').open('a');self.handles.append(f)
  args=[exe,'-u',str(R/'code'/(mode+'.py'))]+(['--run-id',run_id] if run_id else [])
  p=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);self.children.append(p);return p
 def health(self):
  assert time.monotonic()-self.start<self.c['max_runtime_hours']*3600,'Bounded MB1 deadline reached';website();rr=[]
  for spec in self.c['runs']:
   out=R/'runs'/spec['run_id'];p=out/'progress.json';progress=read(p) if p.exists() else None
   rr.append({'run_id':spec['run_id'],'model':spec['model'],'seed':spec['seed'],'inferred_entries':len(list(out.glob('*_inference.json'))),'inference_complete':(out/'inference_complete.json').exists(),'scoring_complete':(out/'metrics.json').exists(),'inference_progress':progress,'score_progress':read(out/'score_progress.json') if (out/'score_progress.json').exists() else None})
  atomic(R/'STATUS.json',{'status':'running','phase':self.phase,'seconds':time.monotonic()-self.start,'runs':rr,'expected_runs':8,'datasets':7,'expected_entries_per_run':106,'expected_planes_per_run':2595,'expected_metric_rows':20760,'physical_gpus':[1,2,3],'gpu0_used':False,'website_http':200,'child_pids':[p.pid for p in self.children if p.poll() is None]})
 def run(self):
  order=['P0','B0','O','M','P1','B1','P2','B2'];pending=[i for i in order if not (R/'runs'/i/'inference_complete.json').exists()];active={};scorers={};done_score={i for i in order if (R/'runs'/i/'metrics.json').exists()}
  while pending or active or len(done_score)<8:
   for gpu,(rid,p) in list(active.items()):
    code=p.poll()
    if code is None:continue
    if code:raise RuntimeError(f'MB1 own inference {rid} failed: {code}; preserve outputs, no protocol search')
    del active[gpu]
   for rid,p in list(scorers.items()):
    code=p.poll()
    if code is None:continue
    if code:raise RuntimeError(f'MB1 own scoring {rid} failed: {code}; inspect log')
    done_score.add(rid);del scorers[rid]
   for gpu in [1,2,3]:
    if gpu not in active and pending:
     rid=pending.pop(0);active[gpu]=(rid,self.launch(rid,'infer',gpu))
   for rid in order:
    if len(scorers)>=2:break
    if rid not in done_score and rid not in scorers and (R/'runs'/rid/'inference_complete.json').exists():scorers[rid]=self.launch(rid,'score')
   self.health();time.sleep(5)
  self.phase='report';p=self.launch(None,'report')
  while p.poll() is None:self.health();time.sleep(5)
  if p.returncode:raise RuntimeError('MB1 final report failed; all measured inference/metrics preserved')
  after=protect();assert after==self.before,'Protected historic assets changed';atomic(R/'evidence/historical_inputs_after.json',after);website()
  atomic(R/'STATUS.json',{'status':'complete','models':4,'runs':8,'datasets':7,'entries_per_run':106,'planes_per_run':2595,'metric_rows':20760,'gpu0_used':False,'website_http':200,'historical_inputs_unchanged':True,'checkpoint_writes':False,'training':False,'further_experiments_started':False,'report':'MB1_summary.md'})
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
 R.mkdir(exist_ok=True);lock=(R/'RUN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ctl=None
 try:ctl=Controller();ctl.run()
 except BaseException as e:atomic(R/'STATUS.json',{'status':'failed','phase':ctl.phase if ctl else 'initialize','error':repr(e),'further_experiments_started':False});raise
 finally:
  if ctl:ctl.stop_own()
