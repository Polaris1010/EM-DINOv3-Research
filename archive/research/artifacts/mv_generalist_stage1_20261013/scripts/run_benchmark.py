"""Guarded smoke -> one common-budget pilot -> 9 matched frozen head runs -> metrics."""
from pathlib import Path
import os,sys,json,time,csv,subprocess,signal,urllib.request,tarfile,collections
import psutil
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mv_data import manifest,sha,atomic
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
def website():
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8) as res:assert res.status==200
def gpus():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(r[0]):{'uuid':r[1].strip(),'free_mib':int(r[2])} for r in csv.reader(raw.splitlines())}
def apps():
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
 return [{'uuid':r[0].strip(),'pid':int(r[1])} for r in csv.reader(raw.splitlines()) if len(r)==3]
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,json.JSONDecodeError):return None
def worker(model):
 p=read(R/'MV_frozen_probe_protocol.yaml');assert p['formal_seed_runs_may_start'] and p['locked_steps']
 for seed in [int(os.environ['MV_DECODER_SEED'])]:
  out=R/'runs'/model/f'seed{seed}';assert not out.exists()
  subprocess.run([PY,'-u',str(R/'src/mv_probe.py'),'--model',model,'--seed',str(seed),'--steps',str(p['locked_steps']),'--out',str(out)],check=True)
def main():
 m=manifest();p=read(R/'MV_frozen_probe_protocol.yaml');assert p['status']=='budget_unlocked_until_pilot'
 registry=read(R/'model_registry.json')['models'];gpu=gpus();assert all(gpu[i]['free_mib']>16000 for i in [1,2,3]);website()
 allowed={}
 for a in apps():
  if a['uuid'] in [gpu[i]['uuid'] for i in [1,2,3]]:
   proc=psutil.Process(a['pid']);assert '/opt/data/3_117/tanghr/mira_superhuman_demo_20260825/venv/bin/python' in proc.cmdline()
   allowed[a['pid']]=proc.create_time()
 protected=[Path(x['checkpoint_path']) for x in registry]+[Path(registry[0]['resolved_config_path']),
 Path('/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/dinov3/models/primus.py'),
 Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/data/pretrain_all_mitoverse.json'),
 Path('/opt/data/3_117/tanghr/MitoVerse_data/catalog.json')]
 protected+=list(Path('/opt/data/3_117/tanghr/MitoVerse_data/splits').glob('*.json'))
 for r in m['entries']:protected.extend(Path(r['image_path'])/k for k in r['metadata_sha256'])
 before={str(f):sha(f) for f in protected};atomic(R/'evidence/protected_before.json',before)
 children=[];logs=[]
 def launch(args,index,name,extra_env=None):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu[index]['uuid'],'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
  env.update(extra_env or {})
  f=(R/'logs'/f'{name}.log').open('w');logs.append(f)
  proc=subprocess.Popen([PY,'-u',*args],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(proc);return proc
 def monitor(procs,stage,folders):
  while True:
   website()
   status=[]
   for proc,folder in zip(procs,folders):status.append({'pid':proc.pid,'exit_code':proc.poll(),'output':str(folder),'progress':read(folder/'progress.json'),'eval_progress':read(folder/'evaluation_progress.json')})
   atomic(R/'STATUS.json',{'status':'running','stage':stage,'workers':status,'time':time.time()})
   for a in apps():
    if a['uuid'] not in [gpu[i]['uuid'] for i in [1,2,3]]:continue
    proc=psutil.Process(a['pid']);known=a['pid'] in allowed and proc.create_time()==allowed[a['pid']]
    own=any(a['pid']==c.pid or any(parent.pid==c.pid for parent in proc.parents()) for c in children)
    if not known and not own:raise RuntimeError('New external GPU process; stopping only own workers')
   if any(c.poll() not in (None,0) for c in procs):raise RuntimeError('MV worker failed; inspect logs')
   if all(c.poll()==0 for c in procs):break
   time.sleep(10)
 try:
  (R/'logs').mkdir(exist_ok=True);(R/'runs').mkdir(exist_ok=True)
  # Real three-source training and held-out validation smoke for all encoders before formal budget pilot.
  procs=[];folders=[]
  for i,model in zip([1,2,3],['P0','B100k','B150k']):
   folder=R/'smoke'/model;folders.append(folder)
   procs.append(launch([str(R/'src/mv_probe.py'),'--model',model,'--out',str(folder),'--smoke'],i,f'smoke_{model}'))
  monitor(procs,'multi_source_smoke',folders)
  receipts=[read(f/'smoke.json') for f in folders];assert all(r['status']=='PASS' for r in receipts)
  assert len({r['head_initial_sha256'] for r in receipts})==1
  atomic(R/'evidence/multi_source_smoke.json',{'status':'PASS','models':receipts})
  pilot=R/'pilot/P0';c=launch([str(R/'src/mv_probe.py'),'--model','P0','--out',str(pilot),'--pilot','--steps',str(p['budget_pilot_steps'])],1,'budget_pilot_P0')
  monitor([c],'common_budget_pilot',[pilot]);pilot_meta=read(pilot/'metrics.json');curve=read(pilot/'curve.json')
  steps=pilot_meta['completed_steps'];assert steps in p['budget_pilot_checkpoints']
  p.update({'status':'locked_after_P0_pilot','locked_steps':steps,'formal_seed_runs_may_start':True,
   'pilot_curve_sha256':sha(pilot/'curve.json'),'budget_selection_model':'P0',
   'budget_plateau_reached':steps<p['budget_pilot_steps'],'maximum_budget_reached':steps==p['budget_pilot_steps']})
  atomic(R/'MV_frozen_probe_protocol.yaml',p);atomic(R/'evidence/budget_lock.json',{'steps':steps,'rule':p['budget_rule'],'curve':curve,'all_models_same_budget':True})
  # Stage B starts only now; pilot head is NOT recycled for P0.
  for j,seed in enumerate(p['seed_sequence']):
   procs=[];folders=[]
   for i,model in zip([1,2,3],['P0','B100k','B150k']):
    folders.append(R/'runs'/model/f'seed{seed}')
    procs.append(launch([__file__,'--worker',model],i,f'formal_{model}_seed{seed}',{'MV_DECODER_SEED':str(seed)}))
   monitor(procs,f'frozen_benchmark_seed{seed}',folders)
   seed_runs=[read(f/'metrics.json') for f in folders]
   assert len({v['samples_sequence_sha256'] for v in seed_runs})==1
   assert len({v['decoder_initial_sha256'] for v in seed_runs})==1
   if j==0:
    atomic(R/'STATUS.json',{'status':'scoring_single_seed','head_training_complete_seed1':True})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    with (R/'logs/single_seed_analysis.log').open('w') as f:
     subprocess.run([PY,'-u',str(R/'scripts/analyze.py'),'--single-seed'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert read(R/'evidence/single_seed_result_audit.json')['status']=='PASS'
  allruns=[r for r in (R/'runs').glob('*/seed*')];assert len(allruns)==9
  for seed in p['seed_sequence']:
   rs=[read(r/'metrics.json') for r in allruns if r.name==f'seed{seed}']
   assert len(rs)==3 and len({v['samples_sequence_sha256'] for v in rs})==1
   assert len({v['decoder_initial_sha256'] for v in rs})==1
   assert all(v['steps']==steps and v['completed_steps']==steps for v in rs)
  atomic(R/'evidence/formal_match_audit.json',{'status':'PASS','groups':9,'initializations_matched_within_seed':True,'crop_sequences_matched_within_seed':True,'locked_steps':steps})
  atomic(R/'STATUS.json',{'status':'scoring','head_training_complete':True,'groups':9})
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
  with (R/'logs/analysis.log').open('w') as f:subprocess.run([PY,'-u',str(R/'scripts/analyze.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  after={str(f):sha(f) for f in protected};assert before==after;website()
  atomic(R/'evidence/protected_integrity.json',{'status':'PASS','before':before,'after':after})
  atomic(R/'STATUS.json',{'status':'complete','groups':9,'metric_computation_complete':True,'physical_gpus':[1,2,3],'website_http_after':200})
  with tarfile.open(R/'delivery.tar.gz','w:gz') as tar:
   for f in R.rglob('*'):
    if not f.is_file() or any(k in f.parts for k in ['cache','__pycache__']) or f.suffix in {'.npy','.pth','.gz'}:continue
    tar.add(f,arcname=str(f.relative_to(R)))
  print('MV_FROZEN_BENCHMARK_COMPLETE',flush=True)
 except BaseException as e:
  for c in children:
   if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
  atomic(R/'STATUS.json',{'status':'failed','error':repr(e),'own_workers_only_stopped':True});raise
 finally:
  for f in logs:f.close()
if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2])
 else:main()
