"""Bounded C-1 only; no launch path for15k SSL or downstream training."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib,urllib.request
R=Path(os.environ['PILOT_C1_ROOT']);A=R.parent.parent;PY=A/'dinov3_shared/bin/python'
def atomic(name,v):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def health():
 assert urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10).status==200
def mapping():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 return {int(i):{'uuid':u.strip(),'memory':int(m),'util':int(v)} for i,u,m,v in (x.split(',') for x in lines)}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 assert not (R/'STATUS.json').exists(),'Do not duplicate preflight'
 protocol=json.loads((R/'protocol.json').read_text());assert protocol['max_updates_per_branch']==20 and not protocol['formal_15000_SSL_authorized'] and not protocol['Primus_authorized']
 health();m=mapping();assert all(m[i]['memory']<1024 and m[i]['util']==0 for i in [1,2,3]),'All three permitted devices must be idle'
 assert sha(R.parent/'ssl_diversity_d12_100k_20261018/code/train_ssl.py')==protocol['legacy_trainer_sha256']
 for rel,digest in [(protocol['parent_full_checkpoint_relative'],protocol['parent_full_checkpoint_sha256']),(protocol['legacy_base_config_relative'],protocol['legacy_base_config_sha256'])]:assert sha(A/rel)==digest
 ema=A/protocol['parent_full_checkpoint_relative'];ema=ema.with_name('teacher_backbone.pth');assert sha(ema)==protocol['anchor_EMA_sha256']
 started=time.monotonic();base={**os.environ,'D12_ROOT':str(R.parent/'ssl_diversity_d12_100k_20261018'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(R/'cache/cuda'),'PYTHONPATH':str(R.parent/'mitotopo_20260908_round2/deps')+':'+str(A/'DINOv3/dinov3')};jobs=[];times={}
 try:
  for label,args,devices in [('unit_tests',[str(PY),str(R/'code/test_barrier.py')],[3]),*[(b,[str(PY),'-m','torch.distributed.run','--standalone','--nproc_per_node=3',str(R/'code/smoke.py'),'--branch',b],[1,2,3]) for b in ['C0','C1']]]:
   if os.environ.get('PILOT_C1_REUSE_C0')=='1' and label in ['unit_tests','C0']:
    completed=[json.loads((R/f'smoke/C0/complete_rank{i}.json').read_text()) for i in range(3)]
    assert all(x['status']=='complete' and x['updates']==20 and x['last_step']==30020 for x in completed)
    times[label+'_reused_no_updates']=True
    continue
   health();live=mapping();assert all(live[i]['memory']<1024 and live[i]['util']==0 for i in devices)
   env={**base,'CUDA_VISIBLE_DEVICES':','.join(m[i]['uuid'] for i in devices),'PILOT_C1_TEST_CUDA':'1'}
   before=time.monotonic()
   with (R/'logs'/f'{label}.log').open('w') as log:
    p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(p)
    while p.poll() is None:
     health();assert time.monotonic()-started<1800,'C-1 bounded runtime expired'
     atomic('STATUS.json',{'status':'running','phase':label,'seconds':time.monotonic()-started,'max_updates_per_branch':20,'formal_SSL_started':False,'Primus_started':False,'GPU0_used':False});time.sleep(5)
    assert p.returncode==0,f'{label} failed; no automatic altered experiment'
   times[label]=time.monotonic()-before
  assert sha(A/protocol['parent_full_checkpoint_relative'])==protocol['parent_full_checkpoint_sha256'] and sha(ema)==protocol['anchor_EMA_sha256']
  atomic('STATUS.json',{'status':'complete','C0_updates':20,'C1_updates':20,'unit_tests_passed':True,'seconds':time.monotonic()-started,'stage_wall_seconds':times,'GPU0_used':False,'website_http':200,'historical_checkpoints_unchanged':True,'formal_SSL_started':False,'Primus_started':False,'further_experiments_started':False})
 except BaseException as e:
  atomic('STATUS.json',{'status':'failed','error':repr(e),'formal_SSL_started':False,'Primus_started':False,'further_experiments_started':False});raise
 finally:
  for p in jobs:
   if p.poll() is None:
    import signal
    os.killpg(p.pid,signal.SIGTERM)
if __name__=='__main__':main()
