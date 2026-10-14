"""Bounded MV3 execution: two encoders x three seeds, then report and stop."""
import argparse,csv,fcntl,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
import psutil
from data import R,OLD,MV2,read,sha,atomic,manifest
from scoring_queue import score_queue
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'

def gpu_info():
 return {int(r[0]):{'uuid':r[1].strip(),'free_mib':int(r[2])} for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
def website():
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10) as f:assert f.status==200
def worker(model,seed):
 out=R/'runs'/model/f'seed{seed}'
 if not (out/'training_complete.json').exists():subprocess.run([PY,'-u',str(R/'src/train.py'),'train','--model',model,'--seed',str(seed)],check=True)
 if not (out/'inference_complete.json').exists():subprocess.run([PY,'-u',str(R/'src/evaluate.py'),'infer','--model',model,'--seed',str(seed)],check=True)

class Controller:
 def __init__(self):
  self.gpus=gpu_info();assert all(self.gpus[i]['free_mib']>18000 for i in [1,2,3]);website();self.children=[];self.handles=[];self.started=time.time();self.phase='initialize'
  self.baseline={}
  allowed={self.gpus[i]['uuid'] for i in [1,2,3]}
  for row in csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()):
   if row[0].strip() not in allowed:continue
   pid=int(row[1])
   try:self.baseline[pid]=psutil.Process(pid).create_time()
   except psutil.NoSuchProcess:self.baseline[pid]=None
  self.protected={str(p):sha(p) for p in [OLD/'MV-Dev-v1.json',OLD/'MV_frozen_probe_protocol.yaml',OLD/'model_registry.json',OLD/'src/mv_probe.py',OLD/'src/official_primus.py',MV2/'MV_dev_validation_manifest.json',MV2/'STATUS.json',MV2/'metrics_all_complete.json']}
  atomic(R/'evidence/historical_inputs_before.json',self.protected)
 def launch(self,args,name,gpu=None):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':self.gpus[gpu]['uuid'] if gpu is not None else '', 'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','MV3_ROOT':str(R),'CUPY_CACHE_DIR':str(R/'cache/cupy')}
  f=(R/'logs'/f'{name}.log').open('a');self.handles.append(f)
  p=subprocess.Popen([PY,'-u',*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);self.children.append(p);return p
 def health(self):
  website();ownroots={p.pid for p in self.children}
  allowed={self.gpus[i]['uuid'] for i in [1,2,3]}
  for row in csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()):
   if row[0].strip() not in allowed:continue
   pid=int(row[1])
   if pid in self.baseline and self.baseline[pid] is None:continue
   try:proc=psutil.Process(pid);known=pid in self.baseline and self.baseline[pid]==proc.create_time();own=pid in ownroots or any(p.pid in ownroots for p in proc.parents())
   except psutil.NoSuchProcess:continue
   if not (known or own):raise RuntimeError('New unrelated process on a permitted GPU; stop only own jobs')
  progress={}
  for p in R.glob('runs/*/seed*/progress.json'):
   try:progress[str(p.relative_to(R))]=read(p)
   except (OSError,ValueError):pass
  atomic(R/'STATUS.json',{'status':'running','phase':self.phase,'seconds':time.time()-self.started,'physical_gpus':[1,2,3],'gpu0_used':False,'website_http':200,'progress':progress,'metrics_completed':len(list(R.glob('runs/*/seed*/*_metrics.json'))),'child_pids':[p.pid for p in self.children if p.poll() is None]})
 def wait(self,children):
  while True:
   self.health()
   for p in children:
    if p.poll() not in [None,0]:raise RuntimeError(f'Own MV3 process {p.pid} exited {p.returncode}; read its log')
   if all(p.poll()==0 for p in children):return
   time.sleep(15)
 def metrics(self,seeds):
  self.phase='full_development_metrics'
  expected={r['id'] for r in manifest()['entries'] if r['supervised_role']=='dev'}
  tasks=[]
  for seed in seeds:
   for model in ['G-P0','G-B100']:
    out=R/'runs'/model/f'seed{seed}';paths=list(out.glob('*_overlap.json'));assert {read(p)['region_id'] for p in paths}==expected
    for path in sorted(paths,key=lambda p:read(p)['valid_voxels']):
     if path.with_name(path.name.replace('_overlap.json','_metrics.json')).exists():continue
     tasks.append({'path':path,'model':model,'seed':seed,'memory_bytes':45*read(path)['valid_voxels']+(4<<30)})
  def launch_score(task,gpu):
   path=task['path'];return self.launch([R/'src/evaluate.py','score','--path',path],f"metric_{task['model']}_{task['seed']}_{path.stem}",gpu)
  # One worker per allowed GPU; conservatively reserve all active jobs' estimated RAM.
  score_queue(tasks,launch_score,self.health,lambda:psutil.virtual_memory().available)
  for seed in seeds:
   for model in ['G-P0','G-B100']:
    out=R/'runs'/model/f'seed{seed}'
    actual=[read(p) for p in out.glob('*_metrics.json')];assert {p['region_id'] for p in actual}==expected
    atomic(out/'validation_complete.json',{'status':'complete','regions':len(actual),'sources':len({p['source_id'] for p in actual}),'tta':False,'manifest_sha256':sha(R/'MV-Generalist.json')})
 def matched(self,seed):
  a,b=[read(R/'runs'/m/f'seed{seed}'/'training_complete.json') for m in ['G-P0','G-B100']]
  assert a['decoder_initial_sha256']==b['decoder_initial_sha256'] and a['sampling_hash']==b['sampling_hash'] and a['source_draws']==b['source_draws']
  assert a['completed_steps']==b['completed_steps']==40000 and a['encoder_unchanged'] and b['encoder_unchanged']
  atomic(R/'evidence'/f'matched_seed{seed}.json',{'status':'PASS','seed':seed,'decoder_initial_sha256':a['decoder_initial_sha256'],'sampling_hash':a['sampling_hash'],'source_draws':a['source_draws'],'steps_each':40000,'manifest_sha256':a['manifest_sha256'],'protocol_sha256':a['protocol_sha256']})
 def run(self):
  protocol=read(R/'MV3_protocol.json');seeds=protocol['seed_sequence'];assert len(seeds)==3
  if all((R/'runs'/m/f'seed{s}'/name).exists() for m in ['G-P0','G-B100'] for s in seeds for name in ['training_complete.json','inference_complete.json']):
   assert read(R/'cache_complete.json')['status']=='PASS'
   for seed in seeds:self.matched(seed)
   self.metrics(seeds);self.finish_report();return
  self.phase='prepare_supervision'
  while not (R/'cache_complete.json').exists():
   service=subprocess.check_output(['systemctl','show',os.environ.get('MV3_PREPARE_SERVICE','mv3-prepare-20261014.service'),'-p','ActiveState','-p','Result'],text=True)
   if 'ActiveState=failed' in service:raise RuntimeError('MV3 cache preparation failed; inspect cache.log')
   self.health();time.sleep(10)
  assert read(R/'cache_complete.json')['status']=='PASS'
  self.phase='smoke_first_seed'
  checks=[]
  for gpu,model in zip([1,2],['G-P0','G-B100']):
   receipt=R/'smoke'/model/f'seed{seeds[0]}'/'smoke.json'
   if not receipt.exists():checks.append(self.launch([R/'src/train.py','smoke','--model',model,'--seed',seeds[0]],f'smoke_{model}',gpu))
  self.wait(checks)
  a,b=[read(R/'smoke'/m/f'seed{seeds[0]}'/'smoke.json') for m in ['G-P0','G-B100']]
  assert a['status']==b['status']=='PASS' and a['decoder_initial_sha256']==b['decoder_initial_sha256'] and a['sampling_sha256']==b['sampling_sha256']
  atomic(R/'evidence/smoke_pair.json',{'status':'PASS','seed':seeds[0],'matched_initialization':True,'matched_sampling':True})
  self.phase='first_seed_train_and_infer'
  jobs=[self.launch([__file__,'worker','--model',m,'--seed',seeds[0]],f'first_{m}',g) for g,m in zip([1,2],['G-P0','G-B100'])];self.wait(jobs);self.matched(seeds[0]);self.metrics([seeds[0]])
  self.phase='first_seed_complete'
  atomic(R/'first_seed_complete.json',{'status':'complete','seed':seeds[0],'models':['G-P0','G-B100'],'sources':12,'regions_per_model':38,'steps_each':40000})
  subprocess.run([PY,str(R/'src/report.py'),'--first-seed'],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
  self.phase='additional_matched_seeds_train_and_infer'
  jobs=[self.launch([__file__,'worker','--model',m,'--seed',s],f'additional_{m}_{s}',g) for g,m,s in [(1,'G-P0',seeds[1]),(2,'G-B100',seeds[1]),(3,'G-P0',seeds[2])]]
  self.wait([jobs[0]])
  jobs.append(self.launch([__file__,'worker','--model','G-B100','--seed',seeds[2]],f'additional_G-B100_{seeds[2]}',1));self.wait(jobs)
  for seed in seeds[1:]:self.matched(seed)
  self.metrics(seeds[1:]);self.finish_report()
 def finish_report(self):
  self.phase='final_report'
  subprocess.run([PY,str(R/'src/report.py')],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
  assert {p:sha(p) for p in self.protected}==self.protected
  atomic(R/'evidence/historical_inputs_after.json',{'status':'PASS','unchanged':self.protected})
  website();atomic(R/'STATUS.json',{'status':'complete','models':2,'decoder_seeds':3,'steps_per_run':40000,'regions_per_run':38,'sources':12,'metric_rows':228,'tta':False,'physical_gpus':[1,2,3],'gpu0_used':False,'website_http_after':200,'report':'MV3_final_report.md','further_experiments_started':False})
 def close(self,failed=False):
  if failed:
   for p in self.children:
    if p.poll() is None:
     try:os.killpg(p.pid,signal.SIGTERM)
     except ProcessLookupError:pass
  for f in self.handles:f.close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['controller','worker']);ap.add_argument('--model');ap.add_argument('--seed',type=int);a=ap.parse_args()
 if a.mode=='worker':worker(a.model,a.seed)
 else:
  lock=(R/'evidence/controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c=Controller()
  try:c.run()
  except Exception:
   import traceback
   atomic(R/'STATUS.json',{'status':'failed','phase':c.phase,'error':traceback.format_exc(),'own_pids':[p.pid for p in c.children]});c.close(True);raise
  else:c.close()
