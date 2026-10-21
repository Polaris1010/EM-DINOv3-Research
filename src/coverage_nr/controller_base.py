"""Bounded matched100k continuation,40k frozen readouts and fixed evaluation; stop."""
import csv,fcntl,os,signal,subprocess,time
from pathlib import Path
import psutil
from common import R,PARENT,M3,OLD,MV2,B_FULL,PY,SSL_PY,WEIGHT,read,atomic,sha,heartbeat
from runtime_guard import website

def protected():
 result={}
 roots=[PARENT,OLD,MV2,M3,B_FULL,R.parent/'mv4_official_mitonet_20261014',R.parent/'mv5_mitonet_adaptation_20261015',R.parent/'mb1_official_benchmark_20261015']
 for root in roots:
  for path in list(root.glob('*'))+list((root/'src').glob('*.py')):
   if path.is_file() and path.suffix in ['.json','.csv','.md','.yaml','.py'] and path.stat().st_size<10_000_000:result[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
 assets=read(R.parent/'mb1_official_benchmark_20261015/MB1_protocol.json')['runs']
 for row in assets:
  for field in ['checkpoint_path','encoder_path']:
   if row.get(field):
    p=Path(row[field]);st=p.stat();result[str(p)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
 for path in PARENT.rglob('*.json'):
  if 'logs' not in path.parts and path.stat().st_size<2_000_000:result[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
 return result

class Controller:
 def __init__(self):
  self.gpus={int(r[0]):{'uuid':r[1].strip(),'free':int(r[2])} for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
  assert all(self.gpus[i]['free']>18000 for i in [1,2,3]);website();self.children=[];self.files=[];self.phase='initialize';self.start=time.monotonic();self.baseline={}
  allowed={self.gpus[i]['uuid'] for i in [1,2,3]}
  for row in csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()):
   if row[0].strip() in allowed:
    try:self.baseline[int(row[1])]=psutil.Process(int(row[1])).create_time()
    except psutil.NoSuchProcess:pass
  self.before=protected();before_path=R/'evidence/historical_inputs_before.json'
  if before_path.exists():assert read(before_path)==self.before,'Historical inputs changed before resume'
  else:atomic(before_path,self.before)
  source_path=R/'evidence/executed_sources.json'
  if source_path.exists():source_path=R/'evidence'/('executed_sources_resume_'+str(time.time_ns())+'.json')
  atomic(source_path,{'files':{str(p.relative_to(R)):sha(p) for p in [*sorted((R/'code').glob('*.py')),R/'D12_protocol.json',R/'D12_base.yaml',R/'D1_manifest.json',R/'D2_manifest.json']},'code_commit':(R/'CODE_COMMIT').read_text().strip(),'SSL_python':str(SSL_PY),'readout_python':str(PY)})
 def launch(self,args,name,gpus,ssl=False):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(self.gpus[g]['uuid'] for g in gpus),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','CUPY_CACHE_DIR':str(R/'cache/cupy'),'NUMBA_CACHE_DIR':str(R/'cache/numba'),'PYTORCH_KERNEL_CACHE_PATH':str(R/'cache/pytorch'),'CUDA_CACHE_PATH':str(R/'cache/cuda'),'PYTHONPATH':('/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps:' if ssl else '')+'/opt/data/3_117/tanghr/DINOv3/dinov3','STAGE1_GUARD_HEARTBEAT':str(R/'HEARTBEAT')}
  f=(R/'logs'/(name+'.log')).open('a');self.files.append(f);p=subprocess.Popen([str(SSL_PY if ssl else PY),*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);self.children.append(p);return p
 def health(self):
  assert time.monotonic()-self.start<72*3600,'Bounded72hour first-round deadline'
  heartbeat(R/'HEARTBEAT',time.time());website();roots={p.pid for p in self.children};allowed={self.gpus[i]['uuid'] for i in [1,2,3]}
  for row in csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()):
   if row[0].strip() not in allowed:continue
   pid=int(row[1])
   try:
    p=psutil.Process(pid);assert (pid in self.baseline and self.baseline[pid]==p.create_time()) or pid in roots or any(v.pid in roots for v in p.parents()),'Unrelated new GPU process; stop only our jobs'
   except psutil.NoSuchProcess:pass
  progress={}
  for path in [*R.glob('runs/*/formal/progress.json'),*R.glob('downstream/runs/*/seed*/progress.json'),*R.glob('benchmark/*/progress.json')]:
   try:progress[str(path.relative_to(R))]=read(path)
   except (OSError,ValueError):pass
  atomic(R/'STATUS.json',{'status':'running','phase':self.phase,'seconds':time.monotonic()-self.start,'progress':progress,'child_pids':[p.pid for p in self.children if p.poll() is None],'physical_gpus':[1,2,3],'gpu0_used':False,'website_http':200,'further_experiments_started':False})
 def wait(self,children):
  while True:
   self.health()
   for p in children:
    if p.poll() not in [None,0]:raise RuntimeError(f'Own child {p.pid} failed:{p.returncode}; inspect phase log, no automatic parameter changes')
   if all(p.poll()==0 for p in children):return
   time.sleep(15)
 def run(self):
  from readout_inputs import prepare
  atomic(R/'evidence/readout_inputs_preflight.json',prepare(R,PARENT))
  assert read(PARENT/'STATUS.json')['status']=='complete'
  assert read(PARENT/'loader_smoke.json')['status']=='PASS'
  for model in ['D1','D2']:
   result_path=R/'runs'/model/'formal/result.json'
   if result_path.exists() and read(result_path)['status']=='complete':
    assert read(result_path)['completed_steps']==100000;continue
   self.phase='SSL_'+model+'_100000';self.health()
   args=['-m','torch.distributed.run','--standalone','--nproc_per_node=3',R/'code/train_ssl.py','--manifest',R/(model+'_manifest.json'),'--config',R/'D12_base.yaml','--weight',WEIGHT,'--output',R/'runs'/model/'formal','--steps','100000','--seed','0']
   latest=R/'runs'/model/'formal/latest_checkpoint.json'
   if latest.exists():
    checkpoint=read(latest);assert 30000<checkpoint['step']<100000
   else:
    parent_dir=PARENT/'runs'/model/'formal'
    meta=read(parent_dir/'checkpoints/step000030000/metadata.json')
    checkpoint={'step':30000,'path':str(parent_dir/'checkpoints/step000030000/full.pth'),'sha256':meta['sha256']}
   from collections import Counter
   counts=Counter()
   parent_dir=PARENT/'runs'/model/'formal'
   for rank in range(3):counts.update(read(parent_dir/f'sampling_totals_rank{rank}.json'))
   prefix=parent_dir/'resume_sampling_prefix.json'
   if prefix.exists():counts.update(read(prefix)['counts'])
   assert sum(counts.values())==30000*48
   if checkpoint['step']>30000:
    from sampling import choose
    pool=read(R/'mitoverse_pool.json');cem=read(R/'cem_pool.json');fraction=read(R/(model+'_manifest.json'))['cem_fraction']
    for i in range(30000*48,checkpoint['step']*48):
     where=choose(pool,cem,fraction,0,i);counts[where['domain']+'::'+where['source']]+=1
   atomic(R/'runs'/model/'formal/resume_sampling_prefix.json',{'checkpoint':checkpoint,'counts':dict(counts),'prefix_includes_original30000':True})
   args+=['--resume',checkpoint['path']]
   self.wait([self.launch(args,'SSL_'+model,[1,2,3],ssl=True)])
   assert read(R/'runs'/model/'formal/result.json')['completed_steps']==100000
  for mode in ['train','infer']:
   self.phase='matched_frozen_head_'+mode
   self.wait([self.launch([R/'code/downstream.py',mode,'--model',model],'head_'+mode+'_'+model,[gpu]) for model,gpu in [('D1',1),('D2',2)]])
  a=read(R/'downstream/runs/D1/seed20261014/training_complete.json');b=read(R/'downstream/runs/D2/seed20261014/training_complete.json')
  assert a['decoder_initial_sha256']==b['decoder_initial_sha256'] and a['sampling_hash']==b['sampling_hash'] and a['encoder_unchanged'] and b['encoder_unchanged']
  atomic(R/'matched_decoder_seed.json',{'status':'PASS','seed':20261014,'decoder_initial_sha256':a['decoder_initial_sha256'],'sampling_hash':a['sampling_hash'],'encoder_frozen':True})
  self.phase='all38_development_metrics';pending=[]
  for model in ['D1','D2']:
   paths=list((R/'downstream/runs'/model/'seed20261014').glob('*_overlap.json'));assert len(paths)==38
   pending+=sorted(paths,key=lambda p:read(p)['valid_voxels'])
  active={}
  while pending or active:
   for gpu,(process,need) in list(active.items()):
    status=process.poll()
    if status is not None:
     assert status==0,f'Own scoring failure {process.pid}';del active[gpu]
   for gpu in [1,2,3]:
    if gpu in active or not pending:continue
    need=45*read(pending[0])['valid_voxels']+(4<<30)
    if need>psutil.virtual_memory().available-(16<<30)-sum(v[1] for v in active.values()):
     if not active:raise MemoryError('Insufficient RAM for exact existing evaluator')
     break
    path=pending.pop(0);active[gpu]=(self.launch([R/'code/downstream.py','score','--path',path],'score_'+path.parent.parent.name+'_'+path.stem,[gpu]),need)
   self.health();time.sleep(2)
  self.phase='fixed_MB1_development_diagnostic'
  self.wait([self.launch([R/'code/benchmark.py','--model',model],'benchmark_'+model,[gpu]) for model,gpu in [('D1',1),('D2',2)]])
  self.phase='report';self.wait([self.launch([R/'code/report.py'],'report',[])])
  after=protected();assert after==self.before,'Historic input snapshot changed';atomic(R/'evidence/historical_inputs_after.json',after);website()
  atomic(R/'STATUS.json',{'status':'complete','SSL_runs':2,'SSL_steps_each':100000,'head_steps_each':40000,'decoder_seeds':1,'development_metric_rows':76,'benchmark_image_rows':5190,'historical_inputs_unchanged':True,'gpu0_used':False,'website_http':200,'auto100k':False,'autoD3':False,'further_experiments_started':False,'report':'D12_100k_final_report.md'})
 def stop_own(self):
  for p in self.children:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  for p in self.children:
   if p.poll() is None:
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
  for f in self.files:f.close()
if __name__=='__main__':
 with (R/'RUN.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ctl=None
  try:ctl=Controller();ctl.run()
  except BaseException as error:atomic(R/'STATUS.json',{'status':'failed','phase':ctl.phase if ctl else 'initialize','error':repr(error),'further_experiments_started':False});raise
  finally:
   if ctl:ctl.stop_own()
