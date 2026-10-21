"""Finite passive result delivery for the one authorized Coverage-NR pipeline."""
import datetime,json,os,shutil,subprocess,sys
from pathlib import Path

os.environ.pop('SSLKEYLOGFILE',None)
os.environ.update(PYTHONUTF8='1',GIT_CONFIG_COUNT='3',GIT_CONFIG_KEY_0='credential.helper',GIT_CONFIG_VALUE_0='',
 GIT_CONFIG_KEY_1='credential.helper',GIT_CONFIG_VALUE_1='!python E:/mira_website_demo/artifacts/d12_control_20261017/win_git_credential.py',
 GIT_CONFIG_KEY_2='http.sslBackend',GIT_CONFIG_VALUE_2='schannel')
import watch_d12_100k_results as base

ROOT=base.ROOT
base.SCRATCH=ROOT.parent/'artifacts/coverage_nr_20261021'
base.REMOTE='/opt/data/3_117/tanghr/codex/coverage_nr_p0_20261021'
base.BRANCH='codex/coverage-nr-p0-training'

def collect(sftp,status):
 folder=base.SCRATCH/'terminal_collected';folder.mkdir(exist_ok=True)
 names=['STATUS.json','Coverage_NR_summary.json','Coverage_NR_final_report.md',
        'Coverage_NR_per_region_metrics.csv','Coverage_NR_per_source_metrics.csv','Coverage_NR_MB1_per_image.csv',
        'Coverage_NR_MB1_per_dataset.csv','Coverage_NR_checkpoint_hashes.csv','Coverage_NR_vs_historical_D2.csv']
 names += ['evidence/'+n for n in ['historical_inputs_before.json','historical_inputs_after.json','executed_sources.json']]
 names += ['runs/Coverage-NR/formal/'+n for n in ['result.json','initialization.json','progress.json',*[f'sampling_totals_rank{i}.json' for i in range(3)]]]
 names += [f'runs/Coverage-NR/formal/checkpoints/step{s:09d}/teacher_backbone.json' for s in [10000,30000,100000]]
 names += ['downstream/runs/Coverage-NR/seed20261014/'+n for n in ['training_complete.json','inference_complete.json','head_step040000.json','run_protocol.json','curve.json']]
 for name in names:
  try:st=sftp.stat(base.REMOTE+'/'+name)
  except OSError:continue
  assert st.st_size<2_000_000,name
  p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);sftp.get(base.REMOTE+'/'+name,str(p))
 return folder

def publish(folder,status,api):
 if base.git('branch','--show-current')!=base.BRANCH or base.git('status','--porcelain'):
  return {'status':'COLLECTED_CHECKOUT_CHANGED_OR_DIRTY','folder':str(folder),'server_status':status}
 before=json.loads((ROOT/'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:57]
 for p in folder.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(folder)
  dest=(ROOT/'docs/Coverage_NR_final_report.md') if p.name=='Coverage_NR_final_report.md' else ROOT/'experiments/Coverage-NR'/rel
  dest.parent.mkdir(parents=True,exist_ok=True)
  # Keep source receipts intact; normalize generated result text for Git.
  dest.write_text(p.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
 ip=ROOT/'experiments/Coverage-NR/run_index.json';idx=json.loads(ip.read_text(encoding='utf-8'))
 idx.update(status='VERIFIED_COMPLETED' if status['status']=='complete' else 'VERIFIED_FAILED',execution_state='COMPLETE_STOPPED' if status['status']=='complete' else 'FAILED_STOPPED',status_evidence='experiments/Coverage-NR/STATUS.json')
 for run in idx['runs']:
  ssl=run['kind']=='SSL'
  relative='runs/Coverage-NR/formal/result.json' if ssl else 'downstream/runs/Coverage-NR/seed20261014/training_complete.json'
  rp=ROOT/'experiments/Coverage-NR'/relative
  if rp.exists():
   rr=json.loads(rp.read_text());run['completed_steps']=rr.get('completed_steps',0)
   run['status']='VERIFIED_COMPLETED' if rr.get('status')=='complete' else 'VERIFIED_FAILED'
  elif run['status']=='VERIFIED_RUNNING':run['status']='VERIFIED_FAILED'
  if run['status']=='VERIFIED_COMPLETED':
   sp=ROOT/'experiments/Coverage-NR'/('runs/Coverage-NR/formal/checkpoints/step000100000/teacher_backbone.json' if ssl else 'downstream/runs/Coverage-NR/seed20261014/head_step040000.json')
   meta=json.loads(sp.read_text());run.update(checkpoint_path=meta['path'],checkpoint_sha256=meta['sha256'])
   if not ssl:
    meta=json.loads((ROOT/'experiments/Coverage-NR/runs/Coverage-NR/formal/checkpoints/step000100000/teacher_backbone.json').read_text());run.update(encoder_path=meta['path'],encoder_sha256=meta['sha256'])
 if status['status']=='complete':idx.update(metrics_path='experiments/Coverage-NR/Coverage_NR_per_region_metrics.csv',report_path='docs/Coverage_NR_final_report.md')
 ip.write_text(json.dumps(idx,indent=2)+'\n',encoding='utf-8',newline='\n')
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 note='Coverage-NR终止状态 '+status['status']+'，UTC '+stamp+'；详见experiments/Coverage-NR/STATUS.json。'
 if status['status']=='complete':note+=' P0连续100k、首seed Primus40k、38dev与2595平面MB1均完成，先读docs/Coverage_NR_final_report.md。'
 else:note+=' 保留实际完成产物与失败阶段，不自动改参或重启。'
 note+=' 本轮停止，不自动新数据、新loss、R组或额外seed。'
 for name in ['AGENTS.md','START_HERE.md','docs/NEW_WINDOW_HANDOFF.md','docs/PROJECT_STATE.md','docs/NEXT_ACTION.md']:
  p=ROOT/name;p.write_text('> '+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
 p=ROOT/'docs/work_logs/20261021_27_coverage_nr_training_start.md';p.write_text(p.read_text(encoding='utf-8')+'\n## 自动终态收取\n\n'+note+'\n',encoding='utf-8',newline='\n')
 p=ROOT/'docs/Coverage_NR_PR.md';p.write_text(note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
 for args in [['scripts/update_registry.py'],['scripts/update_registry.py','--check'],['scripts/validate_repository.py'],['-m','unittest','discover','-s','tests','-q']]:
  subprocess.run([sys.executable,*args],cwd=ROOT,env=base.ENV,check=True)
 assert json.loads((ROOT/'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:57]==before
 base.git('add','AGENTS.md','START_HERE.md','docs','experiments/Coverage-NR','experiments/registry.yaml')
 base.git('commit','-m','Collect actual Coverage-NR terminal results and handoff')
 base.git('-c','http.sslBackend=schannel','-c','http.version=HTTP/1.1','push','origin',base.BRANCH)
 head=base.git('rev-parse','HEAD')
 prs=api('/repos/Polaris1010/EM-DINOv3-Research/pulls?state=open&head=Polaris1010:'+base.BRANCH);assert prs
 pr=api('/repos/Polaris1010/EM-DINOv3-Research/pulls/'+str(prs[0]['number']),'PATCH',{'title':'Coverage-NR: P0 full-coverage100k — '+status['status'],'body':(ROOT/'docs/Coverage_NR_PR.md').read_text(encoding='utf-8')})
 assert pr['draft'] and not pr['merged'] and pr['head']['sha']==head
 return {'status':'PUSHED','commit':head,'pr':pr['html_url'],'server_status':status['status'],'starts_experiments':False}

base.collect=collect
base.publish=publish
if __name__=='__main__':
 try:base.main()
 except BaseException as e:
  (base.SCRATCH/'collector_error.json').write_text(json.dumps({'error':repr(e),'starts_experiments':False})+'\n')
  raise
