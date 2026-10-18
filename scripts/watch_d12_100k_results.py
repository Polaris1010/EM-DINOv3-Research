"""Finite collector for the explicitly authorized D1/D2 pipeline; starts no experiments.

SSH password is entered through getpass, held only in memory, and never saved.
Waits up to72h; downloads small text outputs, checks, commits/pushes this branch,
updates its draft PR and exits. A changed/dirty checkout preserves scratch files.
"""
import argparse,datetime,getpass,json,os,subprocess,sys,time
from pathlib import Path
import paramiko
from watch_mv3_results import verify_private_remote
ROOT=Path(__file__).resolve().parents[1]
SCRATCH=ROOT.parent/'artifacts/d12_100k_control_20261018'
REMOTE='/opt/data/3_117/tanghr/codex/ssl_diversity_d12_100k_20261018'
BRANCH='codex/d12-100k-continuation'
ENV={**os.environ,'PYTHONUTF8':'1','GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never'}
def git(*args):
 p=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'-c','credential.username=Polaris1010',*args],cwd=ROOT,env=ENV,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr[-1800:])
 return p.stdout.strip()
def read(sftp,path):
 with sftp.open(REMOTE+'/'+path) as f:return json.loads(f.read().decode())
def collect(sftp,status):
 folder=SCRATCH/'terminal_collected';folder.mkdir(exist_ok=True)
 def get(remote,relative):
  try:info=sftp.stat(REMOTE+'/'+remote)
  except OSError:return
  assert info.st_size<2_000_000,remote
  local=folder/relative;local.parent.mkdir(parents=True,exist_ok=True);sftp.get(REMOTE+'/'+remote,str(local))
 names=['STATUS.json','D12_100k_summary.json','D12_100k_final_report.md','D12_100k_vs30k.csv','D12_100k_per_region_metrics.csv','D12_100k_source_metrics.csv','D12_100k_paired_deltas.csv','D12_100k_source_deltas.csv','D12_100k_checkpoint_hashes.csv','D12_100k_benchmark_per_dataset.csv','D12_100k_benchmark_per_image.csv','D12_100k_benchmark_historical_first_seed.csv','D12_100k_historical_MV3_first_seed_reference.csv','D12_100k_training_curves_stride100.csv','D12_100k_sampling_observed.json','matched_decoder_seed.json']
 for n in names:get(n,n)
 for n in ['historical_inputs_before.json','historical_inputs_after.json','executed_sources.json']:get('evidence/'+n,'evidence/'+n)
 for model in ['D1','D2']:
  for n in ['result.json','progress.json','initialization.json',*[f'sampling_totals_rank{i}.json' for i in range(3)]]:get(f'runs/{model}/formal/'+n,f'runs/{model}/SSL/'+n)
  for step in [50000,100000]:get(f'runs/{model}/formal/checkpoints/step{step:09d}/teacher_backbone.json',f'runs/{model}/SSL/teacher_step{step:06d}.json')
  for n in ['training_complete.json','inference_complete.json','head_step040000.json','progress.json','run_protocol.json','curve.json']:get(f'downstream/runs/{model}/seed20261014/'+n,f'runs/{model}/HEAD/'+n)
 return folder

def publish(folder,status,api):
 if git('branch','--show-current')!=BRANCH or git('status','--porcelain'):
  return {'status':'COLLECTED_CHECKOUT_CHANGED_OR_DIRTY','folder':str(folder),'server_status':status}
 import shutil
 before=json.loads((ROOT/'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:53]
 for path in folder.rglob('*'):
  if not path.is_file():continue
  rel=path.relative_to(folder)
  target=ROOT/'docs/D12_100k_final_report.md' if rel.as_posix()=='D12_100k_final_report.md' else ROOT/'evidence/D12-100k'/Path(*rel.parts[1:]) if rel.parts[0]=='evidence' else ROOT/'experiments/D12-100k'/rel
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 index=ROOT/'experiments/D12-100k/run_index.json';j=json.loads(index.read_text(encoding='utf-8'));j.update(status='VERIFIED_COMPLETED' if status['status']=='complete' else 'VERIFIED_FAILED',execution_state='COMPLETE_STOPPED' if status['status']=='complete' else 'STOPPED_'+status.get('phase','UNKNOWN'),status_evidence='experiments/D12-100k/STATUS.json',observed_at_utc=stamp)
 for run in j['runs']:
  d=ROOT/'experiments/D12-100k/runs'/run['model']/run['kind'];result=d/('result.json' if run['kind']=='SSL' else 'training_complete.json')
  if result.exists():
   rr=json.loads(result.read_text());run['completed_steps']=rr.get('completed_steps',0);run['status']='VERIFIED_COMPLETED' if rr['status']=='complete' else 'VERIFIED_FAILED'
  elif status['status']=='failed' and run['status']=='VERIFIED_RUNNING':run['status']='VERIFIED_FAILED'
  if run['status']=='VERIFIED_COMPLETED':
   side=d/('teacher_step100000.json' if run['kind']=='SSL' else 'head_step040000.json');meta=json.loads(side.read_text());run.update(checkpoint_path=meta['path'],checkpoint_sha256=meta['sha256'])
   if run['kind']=='HEAD':
    enc=json.loads((d.parent/'SSL/teacher_step100000.json').read_text());run.update(encoder_path=enc['path'],encoder_sha256=enc['sha256'])
 if status['status']=='complete':j.update(metrics_path='experiments/D12-100k/D12_100k_per_region_metrics.csv',report_path='docs/D12_100k_final_report.md')
 index.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8',newline='\n')
 note='D1/D2终止状态 '+status['status']+'，UTC '+stamp+'。'
 if status['status']=='complete':
  s=json.loads((ROOT/'experiments/D12-100k/D12_100k_summary.json').read_text());note+=' 两组累计100k SSL、首seed40k Primus、76条开发指标及5190条固定MB1诊断均完成；开发Dice D1='+format(s['metrics']['D1']['dice'],'.6f')+'，D2='+format(s['metrics']['D2']['dice'],'.6f')+'，D2−D1='+format(s['D2_minus_D1_dev_dice'],'+.6f')+'。先读docs/D12_100k_final_report.md。旧资产前后快照一致，网站200，GPU0未用，不自动更长SSL、D3或更多seed。'
  note+=' MB1七组Dice D1='+format(s['benchmark_diagnostic_macro_dice']['D1'],'.6f')+'，D2='+format(s['benchmark_diagnostic_macro_dice']['D2'],'.6f')+'。内部与跨数据集结论分开，不能仅凭内部均值提升声称广泛泛化。'
 else:note+=' 保留实际产物，查看STATUS与服务器阶段日志；未完成部分不得声称完成，不自动改参/重启。'
 for name in ['AGENTS.md','START_HERE.md','docs/NEW_WINDOW_HANDOFF.md','docs/PROJECT_STATE.md','docs/NEXT_ACTION.md']:
  p=ROOT/name;p.write_text('> '+note+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
 work=ROOT/'docs/work_logs/20261018_18_d12_100k_continuation.md';work.write_text(work.read_text(encoding='utf-8')+'\n## 最终自动收取\n\n'+note+'\n\n本收取器只读取已有结果，不启动实验或修改服务器数据。全部本轮小型文本产物、代码和Markdown一并提交推送，draft PR不合并。\n',encoding='utf-8',newline='\n')
 prfile=ROOT/'docs/D12_100k_PR.md';prfile.write_text(note+'\n\n固定对照变量为50%CEM数据；同自然初始化、同100k、同40k首seed冻结Primus。历史B100预算和暴露不同，不能把D1−B100当纯采样消融；未知CEM别名仍不证明独立。原53条注册与MV1至MV5/MB1保持。\n',encoding='utf-8',newline='\n')
 for args in [['scripts/update_registry.py'],['scripts/validate_d12.py'],['scripts/update_registry.py','--check'],['scripts/validate_repository.py'],['-m','unittest','discover','-s','tests','-q']]:subprocess.run([sys.executable,*args],cwd=ROOT,env=ENV,check=True)
 assert json.loads((ROOT/'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:53]==before
 git('diff','--check');git('add','AGENTS.md','START_HERE.md','docs','evidence/D12-100k','experiments/D12-100k','experiments/registry.yaml');git('commit','-m','Collect actual D1/D2 terminal metrics and research handoff');head=git('rev-parse','HEAD')
 for attempt in range(12):
  try:git('-c','http.sslBackend=schannel','-c','http.version=HTTP/1.1','push','origin',BRANCH);break
  except RuntimeError:
   if attempt==11:raise
   time.sleep(10)
 matches=api('/repos/Polaris1010/EM-DINOv3-Research/pulls?state=open&head=Polaris1010:'+BRANCH)
 assert matches,'Authorized initial draft PR missing'
 pr=api('/repos/Polaris1010/EM-DINOv3-Research/pulls/'+str(matches[0]['number']),'PATCH',{'title':'D1/D2: matched100k continuation — '+status['status'],'body':prfile.read_text(encoding='utf-8')})
 assert pr['draft'] and not pr['merged']
 ref=api('/repos/Polaris1010/EM-DINOv3-Research/git/ref/heads/'+BRANCH);assert ref['object']['sha']==head
 return {'status':'PUSHED','commit':head,'branch':BRANCH,'server_status':status['status'],'pr':pr['html_url'],'draft':True,'further_experiments_started':False}

def main():
 SCRATCH.mkdir(parents=True,exist_ok=True)
 password=getpass.getpass('SSH password (memory only): ');api=verify_private_remote();start=time.monotonic();client=None;connected=False
 try:
  while time.monotonic()-start<72*3600:
   try:
    if client is None or not client.get_transport() or not client.get_transport().is_active():
     client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy());client.connect('192.168.203.97',username='root',password=password,look_for_keys=False,allow_agent=False,timeout=30);client.get_transport().set_keepalive(30)
    with client.open_sftp() as sftp:
     status=read(sftp,'STATUS.json')
     if status['status'] in ['complete','failed']:
      folder=collect(sftp,status);receipt=publish(folder,status,api);(SCRATCH/'terminal_delivery.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True);return
    if not connected:
     (SCRATCH/'collector_connected.json').write_text(json.dumps({'status':'WAITING_FOR_AUTHORIZED_PIPELINE','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'starts_experiments':False}));print('COLLECTOR_WAITING',flush=True);connected=True
   except (paramiko.SSHException,EOFError,OSError,TimeoutError):
    if client:client.close()
    client=None
   time.sleep(30)
  raise TimeoutError('72hour finite collector deadline; preserve server outputs')
 finally:
  if client:client.close()
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if not isinstance(error,KeyboardInterrupt):(SCRATCH/'collector_error.json').write_text(json.dumps({'error':repr(error),'starts_experiments':False})+'\n',encoding='utf-8')
  raise
