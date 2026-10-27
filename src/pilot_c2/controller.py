"""Only the authorized fixed C-2 pipeline; owns/kills only its own child groups."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,urllib.request,signal
R=Path(os.environ['PILOT_C2_ROOT']);A=R.parent.parent;SSL=A/'dinov3_shared/bin/python';PY=A/'envs/meddinov3/bin/python'
MODELS=['C0_Vanilla_45k','C1_Barrier_45k']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def gpus():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 return {int(i):{'uuid':u.strip(),'memory':int(m),'util':int(v)} for i,u,m,v in (x.split(',') for x in lines)}
def website():return urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10).status
class Controller:
 def __init__(self):
  self.start=time.monotonic();self.phase='verify_inputs';self.children=[];self.assignments={};self.logs=[];self.mapping=gpus();self.p=read(R/'protocol.json')
  assert all(self.mapping[i]['memory']<1024 and self.mapping[i]['util']==0 for i in [1,2,3]) and website()==200
  assert self.p['models']==MODELS and self.p['final_step']==45000 and self.p['lambda']==1 and self.p['margin']==.05 and not self.p['diagnostic_VJP']
 def idle(self,i):
  if any(p.poll() is None and i in devices for p,devices in self.assignments.items()):return False
  x=gpus()[i];return x['memory']<1024 and x['util']==0
 def health(self):
  assert time.monotonic()-self.start<18*3600 and website()==200
  assert all(x.poll() in (None,0) for x in self.children),'An owned job failed; preserve existing outputs'
  atomic(R/'STATUS.json',{'status':'running','phase':self.phase,'seconds':time.monotonic()-self.start,'GPU0_used':False,'website_http':200,'fixed_SSL_final_step':45000,'Primus_steps':40000,'no_extra_experiments':True})
 def launch(self,file,args=(),devices=(),model=None,ssl=False):
  assert all(self.idle(i) for i in devices)
  env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(self.mapping[i]['uuid'] for i in devices),'D12_ROOT':str(R.parent/'ssl_diversity_d12_100k_20261018'),'OMP_NUM_THREADS':'4' if ssl else '2','MKL_NUM_THREADS':'4' if ssl else '2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(R/'cache/cuda'),'CUPY_CACHE_DIR':str(R/'cache/cupy'),'PYTORCH_KERNEL_CACHE_PATH':str(R/'cache/pytorch'),'PYTHONPATH':str(R.parent/'mitotopo_20260908_round2/deps')+':'+str(A/'DINOv3/dinov3'),'PILOT_C1_TEST_CUDA':'1'}
  if model:env['PILOT_C2_MODEL']=model
  command=[str(SSL if ssl else PY)]
  if len(devices)==3:command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=3']
  command +=[str(R/'code'/file),*map(str,args)]
  label=file.replace('.py','')+'_'+(model or 'shared')+'_'+str(time.time_ns());f=(R/'logs'/(label+'.log')).open('w');self.logs.append(f)
  job=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);self.children.append(job);self.assignments[job]=list(devices);return job
 def wait(self,jobs):
  while any(x.poll() is None for x in jobs):self.health();time.sleep(10)
  assert all(x.returncode==0 for x in jobs)
 def run(self):
  parent=R.parent/'ssl_diversity_d12_20261017/runs/D2/formal/checkpoints/step000030000/full.pth';ema=parent.with_name('teacher_backbone.pth')
  for p,digest in [(parent,self.p['parent_full_sha256']),(ema,self.p['parent_EMA_sha256']),(R.parent/'ssl_diversity_d12_100k_20261018/D12_base.yaml',self.p['legacy_SSL_config_sha256']),(R.parent/'ssl_diversity_d12_100k_20261018/code/train_ssl.py',self.p['legacy_SSL_trainer_sha256']),(R/'code/barrier.py',self.p['barrier_core_sha256'])]:assert sha(p)==digest
  for model in MODELS:
   system=R/'systems'/model
   for folder in [system,system/'evidence',system/'downstream',system/'linear']:
    folder.mkdir(parents=True,exist_ok=True)
   atomic(system/'protocol.json',self.p)
  self.phase='production_unit_tests';self.wait([self.launch('test_production.py',devices=[3],ssl=True)])
  for model in MODELS:
   self.phase='SSL_'+model;out=R/'runs'/model
   if (out/'result.json').exists():assert read(out/'result.json')['completed_step']==45000;continue
   args=['--model',model]
   if (out/'latest_checkpoint.json').exists():args+=['--resume',read(out/'latest_checkpoint.json')['path']]
   self.wait([self.launch('train_ssl.py',args,devices=[1,2,3],ssl=True)])
  self.phase='verify_complete_SSL_pair';self.wait([self.launch('verify_pair.py',ssl=True)])
  self.phase='matched_Primus_and_Linear'
  heads=[]
  for model,gpu in zip(MODELS,[3,2]):
   done=R/'systems'/model/'downstream/runs'/model/'seed20261014/training_complete.json'
   if done.exists():assert read(done)['completed_steps']==40000
   else:heads.append(self.launch('downstream.py',['train'],devices=[gpu],model=model))
  for model in MODELS:
   if not (R/'systems'/model/'linear/complete.json').exists():self.wait([self.launch('linear.py',devices=[1],model=model)])
  self.wait(heads)
  for model in MODELS:
   done=read(R/'systems'/model/'downstream/runs'/model/'seed20261014/training_complete.json')
   assert done['completed_steps']==40000 and done['encoder_unchanged'] and done['decoder_initial_sha256']==self.p['decoder_initial_sha256'] and done['sampling_hash']==self.p['final_sampling_hash']
  self.phase='all38dev_and7MB1_for_each_model';infer={};bench={};scores={};score_memory={};waiting_MB=MODELS.copy()
  for model,gpu in zip(MODELS,[3,2]):
   out=R/'systems'/model/'downstream/runs'/model/'seed20261014'
   if not (out/'inference_complete.json').exists():infer[model]=self.launch('downstream.py',['infer'],devices=[gpu],model=model)
  while True:
   self.health()
   scores={p:j for p,j in scores.items() if j.poll() is None}
   if waiting_MB:
    model=waiting_MB[0]
    if (R/'systems'/model/'benchmark'/model/'metrics.json').exists():waiting_MB.pop(0)
    else:
     free=[i for i in [1,3,2] if self.idle(i)]
     if free:bench[model]=self.launch('benchmark.py',['--model',model],devices=[free[0]],model=model);waiting_MB.pop(0)
   pending=[]
   for model in MODELS:
    out=R/'systems'/model/'downstream/runs'/model/'seed20261014'
    pending +=[(model,p) for p in sorted(out.glob('*_overlap.json')) if not p.with_name(p.name.replace('_overlap.json','_metrics.json')).exists() and str(p) not in scores]
   for model,path in pending:
    if len(scores)>=2:break
    free=[i for i in [1,3,2] if self.idle(i)]
    if not free:break
    prediction=path.with_name(path.name.replace('_overlap.json','_prediction.npy'))
    estimate=24*prediction.stat().st_size+(2<<30)
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    reserved=sum(score_memory[k] for k in scores)
    if available-(32<<30)-reserved<estimate:continue
    scores[str(path)]=self.launch('downstream.py',['score','--path',path],devices=[free[0]],model=model);score_memory[str(path)]=estimate
   all_done=all((R/'systems'/m/'benchmark'/m/'metrics.json').exists() and len(list((R/'systems'/m/'downstream/runs'/m/'seed20261014').glob('*_metrics.json')))==38 for m in MODELS)
   if all_done and not scores and all(j.poll()==0 for j in [*infer.values(),*bench.values()]):break
   time.sleep(10)
  self.phase='five_system_report';self.wait([self.launch('report.py')])
  assert sha(parent)==self.p['parent_full_sha256'] and sha(ema)==self.p['parent_EMA_sha256']
  atomic(R/'STATUS.json',{'status':'complete','models':MODELS,'additional_SSL_steps_each':15000,'global_final_step':45000,'Primus_steps_each':40000,'Linear_steps_each':500,'dev_regions_each':38,'MB1_images_each':2595,'paired_SSL_verified':True,'GPU0_used':False,'website_http':website(),'historical_checkpoints_unchanged':True,'further_experiments_started':False,'seconds':time.monotonic()-self.start})
if __name__=='__main__':
 import fcntl
 controller=None
 with (R/'RUN.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (R/'STATUS.json').exists(),'No duplicate controller; explicit scoped recovery required'
  try:controller=Controller();controller.run()
  except BaseException as e:atomic(R/'STATUS.json',{'status':'failed','phase':controller.phase if controller else 'initialize','error':repr(e),'further_experiments_started':False});raise
  finally:
   if controller:
    for p in controller.children:
     if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    for f in controller.logs:f.close()
