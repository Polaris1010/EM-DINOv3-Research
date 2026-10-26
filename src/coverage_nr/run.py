"""Authorized single fresh-P0 run, fixed readout/evaluation, then stop."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import psutil
from common import R,PARENT,PY,SSL_PY,WEIGHT,read,atomic,sha
import controller_base as base
from runtime_guard import website

MODEL='Coverage-NR'
old_protected=base.protected
def protected():
 result=old_protected()
 old=R.parent/'ssl_diversity_d12_100k_20261018'
 for p in [*old.glob('*.json'),*old.glob('*.yaml'),*old.glob('*.csv'),*old.glob('code/*.py')]:
  if p.is_file():result[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
 for p in old.glob('runs/*/formal/checkpoints/*/*.pth'):
  st=p.stat();result[str(p)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
 return result
base.protected=protected

class Controller(base.Controller):
 def ssl_args(self,out,steps,resume=None):
  args=['-m','torch.distributed.run','--standalone','--nproc_per_node=3',R/'code/train_ssl.py',
        '--manifest',R/'D2_manifest.json','--config',R/'D12_base.yaml','--weight',WEIGHT,
        '--output',out,'--steps',str(steps),'--seed','0']
  if resume:args+=['--resume',resume]
  return args
 def smoke(self):
  self.phase='smoke_fresh_P0_10_steps';out=R/'runs'/MODEL/'smoke'
  self.wait([self.launch(self.ssl_args(out,10),'smoke_fresh',[1,2,3],ssl=True)])
  a=read(out/'result.json');assert a['status']=='complete' and a['completed_steps']==10
  assert a['max_rank_parameter_difference']<=1e-6 and a['tracked_student_delta_l2']>0
  first=read(out/'initialization.json');assert first['starting_step']==0 and first['resume'] is None
  self.phase='smoke_full_state_resume_12_steps'
  self.wait([self.launch(self.ssl_args(out,12,read(out/'latest_checkpoint.json')['path']),'smoke_resume',[1,2,3],ssl=True)])
  b=read(out/'result.json');assert b['status']=='complete' and b['completed_steps']==12
  assert b['max_rank_parameter_difference']<=1e-6
  atomic(R/'smoke.json',{'status':'PASS','fresh_steps':10,'resumed_step':12,
          'global_batch':48,'physical_gpus':[1,2,3],'GPU0_used':False,'initialization':first,
          'resume_result':b,'plan_verification':read(R/'plan/verification.json')})
  atomic(R/'STATUS.json',{'status':'smoke_complete','phase':'awaiting_formal_launch','training_budget':100000,'gpu0_used':False})
 def run(self):
  assert read(R/'smoke.json')['status']=='PASS'
  assert read(R/'plan/verification.json')['status']=='PASS'
  from readout_inputs import prepare
  atomic(R/'evidence/readout_inputs_preflight.json',prepare(R,PARENT))
  out=R/'runs'/MODEL/'formal';self.phase='SSL_Coverage-NR_fresh_P0_100000';self.health()
  assert not out.exists() or not any(out.iterdir()),'Do not restart an existing run'
  self.wait([self.launch(self.ssl_args(out,100000),'SSL_Coverage-NR',[1,2,3],ssl=True)])
  assert read(out/'result.json')['completed_steps']==100000
  for mode in ['train','infer']:
   self.phase='frozen_Primus_'+mode
   self.wait([self.launch([R/'code/downstream.py',mode,'--model',MODEL],'head_'+mode,[1])])
  complete=read(R/'downstream/runs'/MODEL/'seed20261014/training_complete.json')
  assert complete['completed_steps']==40000 and complete['encoder_unchanged']
  self.phase='all38_development_metrics'
  pending=sorted((R/'downstream/runs'/MODEL/'seed20261014').glob('*_overlap.json'),key=lambda p:read(p)['valid_voxels'])
  assert len(pending)==38
  active={}
  while pending or active:
   for gpu,(process,need) in list(active.items()):
    status=process.poll()
    if status is not None:
     assert status==0,f'Own scoring process failed:{process.pid}';del active[gpu]
   for gpu in [1,2,3]:
    if gpu in active or not pending:continue
    need=45*read(pending[0])['valid_voxels']+(4<<30)
    if need>psutil.virtual_memory().available-(16<<30)-sum(v[1] for v in active.values()):
     if not active:raise MemoryError('Insufficient RAM for existing exact evaluator')
     break
    p=pending.pop(0);active[gpu]=(self.launch([R/'code/downstream.py','score','--path',p],'score_'+p.stem,[gpu]),need)
   self.health();time.sleep(2)
  self.phase='fixed_MB1_development_diagnostic'
  self.wait([self.launch([R/'code/benchmark.py','--model',MODEL],'MB1',[1])])
  self.phase='report'
  self.wait([self.launch([R/'code/report.py'],'report',[])])
  after=protected();assert after==self.before
  atomic(R/'evidence/historical_inputs_after.json',after);website()
  atomic(R/'STATUS.json',{'status':'complete','SSL_steps':100000,'head_steps':40000,
          'development_regions':38,'benchmark_planes':2595,'historical_inputs_unchanged':True,
          'gpu0_used':False,'website_http':200,'further_experiments_started':False,'report':'Coverage_NR_final_report.md'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
 with (R/'RUN.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ctl=None
  try:
   ctl=Controller()
   if a.smoke:ctl.smoke()
   else:ctl.run()
  except BaseException as e:
   atomic(R/'STATUS.json',{'status':'failed','phase':ctl.phase if ctl else 'initialize','error':repr(e),'further_experiments_started':False});raise
  finally:
   if ctl:ctl.stop_own()
