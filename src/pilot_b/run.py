"""One bounded Pilot B. Linear performance never gates the40k Primus run."""
import fcntl,os,subprocess,time,urllib.request
from pathlib import Path
from common import R,PA,MODEL,PY,read,atomic,sha
def gpus():
 t=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(i):{'uuid':u.strip(),'memory_MB':int(m),'util':int(v)} for i,u,m,v in (s.split(',') for s in t.splitlines())}
def website():return urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10).status
class Controller:
 def __init__(self):
  self.start=time.monotonic();self.children=[];self.logs=[];self.stage='initialize';self.mapping=gpus();self.allowed=[i for i in [3,2,1] if self.mapping[i]['memory_MB']<1024 and self.mapping[i]['util']==0]
  assert self.allowed and website()==200,'No idle permitted GPU; do not interfere with unknown jobs'
  self.main=self.allowed[0];self.aux=self.allowed[1] if len(self.allowed)>1 else self.main;self.score=self.allowed[2] if len(self.allowed)>2 else self.main
  atomic(R/'evidence/launch_GPU.json',{'mapping':self.mapping,'roles':{'training_and_dev':self.main,'MB1_and_existing_endpoints':self.aux,'exact_dev_scoring':self.score},'GPU0_used':False,'website_http':200})
 def launch(self,name,args=(),gpu=None):
  if gpu is not None:
   live=gpus()[gpu];assert live['memory_MB']<1024 and live['util']==0,'Assigned GPU no longer idle'
  env={**os.environ,'CUDA_VISIBLE_DEVICES':self.mapping[gpu]['uuid'] if gpu is not None else '',
   'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
   'CUDA_CACHE_PATH':str(R/'cache/cuda'),'CUPY_CACHE_DIR':str(R/'cache/cupy'),'MPLCONFIGDIR':str(R/'cache/mpl')}
  label=name.replace('.py','')+'_'+('_'.join(str(x).replace('/','_') for x in args) or str(time.time_ns()))
  handle=(R/'logs'/(label+'.log')).open('w');self.logs.append(handle)
  p=subprocess.Popen([PY,str(R/'code'/name),*map(str,args)],env=env,stdout=handle,stderr=subprocess.STDOUT);self.children.append(p);return p
 def health(self):
  assert time.monotonic()-self.start<8*3600,'Pilot B8h bound';assert website()==200
  assert all(p.poll() in (None,0) for p in self.children),'An owned child failed'
  atomic(R/'STATUS.json',{'status':'running','phase':self.stage,'seconds':time.monotonic()-self.start,'own_children':[p.pid for p in self.children if p.poll() is None],'GPU0_used':False,'website_http':200,'Linear_AP_is_not_a_Primus_gate':True})
 def wait(self,jobs):
  while any(p.poll() is None for p in jobs):
   self.health()
   for p in jobs:assert p.poll() in (None,0),f'Owned child {p.pid} failed; no automatic altered experiment'
   time.sleep(10)
  assert all(p.returncode==0 for p in jobs)
 def run(self):
  self.stage='construct_single_lerp05';self.wait([self.launch('merge.py',gpu=self.main)])
  self.stage='matched_Linear23_500';self.wait([self.launch('linear.py',gpu=self.main)])
  # No metric value is read here. Every numerically valid midpoint runs the same40k.
  self.stage='Primus40000_and_fixed_endpoint_muscle_AP';train=self.launch('downstream.py',['train'],self.main)
  if self.aux==self.main:self.wait([train])
  for endpoint in ['D2-30k','D2-100k']:
   job=self.launch('benchmark.py',['--model',endpoint],self.aux);self.wait([job])
  self.wait([train])
  out=R/'downstream/runs'/MODEL/'seed20261014';done=read(out/'training_complete.json')
  assert done['completed_steps']==40000 and done['encoder_unchanged']
  assert done['decoder_initial_sha256']==read(R/'protocol.json')['decoder_initial_sha256']
  assert done['sampling_hash']==read(R/'protocol.json')['final_sampling_hash']
  self.stage='38dev_and_fixed7group_MB1'
  infer=self.launch('downstream.py',['infer'],self.main)
  if self.aux==self.main:self.wait([infer])
  benchmark=self.launch('benchmark.py',['--model',MODEL],self.aux);score=None
  while True:
   self.health();assert infer.poll() in (None,0) and benchmark.poll() in (None,0)
   if score is not None and score.poll() is not None:assert score.returncode==0;score=None
   pending=[p for p in sorted(out.glob('*_overlap.json')) if not p.with_name(p.name.replace('_overlap.json','_metrics.json')).exists()]
   if score is None and pending:
    idle=[i for i in self.allowed if gpus()[i]['memory_MB']<1024 and gpus()[i]['util']==0]
    if idle:score=self.launch('downstream.py',['score','--path',pending[0]],idle[-1])
   if infer.poll()==0 and benchmark.poll()==0 and score is None and not pending:break
   time.sleep(10)
  assert len(list(out.glob('*_metrics.json')))==38
  self.stage='aggregate_and_stop';self.wait([self.launch('report.py')])
  for spec in read(R/'protocol.json')['parents'].values():assert sha(spec['path'])==spec['sha256']
  atomic(R/'STATUS.json',{'status':'complete','model':MODEL,'alpha':.5,'Primus_steps':40000,'Linear_steps':500,'dev_regions':38,'MB1_groups':7,'website_http':website(),'GPU0_used':False,'historical_parent_weights_unchanged':True,'new_SSL_steps':0,'Pilot_C_started':False,'further_experiments_started':False,'seconds':time.monotonic()-self.start})
if __name__=='__main__':
 controller=None
 with (R/'RUN.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (R/'STATUS.json').exists(),'Never duplicate Pilot B'
  try:controller=Controller();controller.run()
  except BaseException as e:atomic(R/'STATUS.json',{'status':'failed','phase':controller.stage if controller else 'initialize','error':repr(e),'further_experiments_started':False});raise
  finally:
   if controller:
    for p in controller.children:
     if p.poll() is None:p.terminate()
    for h in controller.logs:h.close()
