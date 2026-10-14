"""Finite passive collection of this authorized MV3 run; never starts experiments.

Credentials arrive on an anonymous stdin pipe, remain in memory, and are not logged.
If the checkout has other changes, preserve collected files in scratch for handoff.
"""
import argparse,datetime,getpass,json,os,subprocess,sys,time
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1]
SCRATCH=ROOT.parent/'artifacts/mv3_control_20261014'
REMOTE='/opt/data/3_117/tanghr/codex/mv_generalist_v2_han_fullgt_20261014'
BRANCH='codex/mv3-generalist-training'
MODELS=['G-P0','G-B100'];SEEDS=[20261014,20261015,20261016]
ENV={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never','PYTHONUTF8':'1'}

def git(*args):
 p=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','-c','credential.username=Polaris1010',*args],cwd=ROOT,env=ENV,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr[-2000:])
 return p.stdout.strip()
def read_remote(sftp,path):
 with sftp.open(REMOTE+'/'+path,'r') as f:return json.loads(f.read().decode())
def download(sftp,path,target):
 target.parent.mkdir(parents=True,exist_ok=True);sftp.get(REMOTE+'/'+path,str(target))
 assert target.stat().st_size<2_000_000,'Unexpected large artifact; do not commit'
def available(sftp,path):
 try:sftp.stat(REMOTE+'/'+path);return True
 except OSError:return False
def clean_checkout():return git('branch','--show-current')==BRANCH and not git('status','--porcelain')

def verify_private_remote():
 import urllib.request
 p=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\nusername=Polaris1010\n\n',capture_output=True,text=True,env=ENV,timeout=20)
 values=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line);token=values.get('password')
 if not token:raise RuntimeError('GitHub authorization unavailable; preserve local results')
 def api(path,method='GET',payload=None):
  data=json.dumps(payload).encode() if payload is not None else None
  request=urllib.request.Request('https://api.github.com'+path,data=data,method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'MV3-passive-results-collector'})
  for attempt in range(6):
   try:
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
   except (ConnectionError,TimeoutError,urllib.error.URLError):
    if attempt==5:raise
    time.sleep(5*(attempt+1))
 repo=api('/repos/Polaris1010/EM-DINOv3-Research');assert repo['private'] and repo['full_name']=='Polaris1010/EM-DINOv3-Research'
 assert api('/user')['login'].lower()=='polaris1010'
 return api

def collect(sftp,status):
 folder=SCRATCH/'collected_v2_results';folder.mkdir(exist_ok=True)
 (folder/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8',newline='\n')
 names=['MV3_final_report.md','MV3_first_seed_report.md','MV3_summary.json','MV3_first_seed_summary.json','MV3_per_region_metrics.csv','MV3_source_metrics.csv','MV3_P0_B100_paired_deltas.csv','MV3_failure_cases.csv','MV3_checkpoint_hashes.csv','MV3_training_curves.csv','MV3_same_region_supervision_gain.csv','MV3_historical_prediction_reference.csv','MV3_historical_reference_unavailable.json','MV3_actual_supervision.json','first_seed_complete.json']
 for name in names:
  if available(sftp,name):download(sftp,name,folder/name)
 for model in MODELS:
  for seed in SEEDS:
   prefix=f'runs/{model}/seed{seed}'
   if not available(sftp,prefix):continue
   for name in sftp.listdir(REMOTE+'/'+prefix):
    if name.endswith('.json'):download(sftp,prefix+'/'+name,folder/prefix/name)
 for name in ['historical_inputs_before.json','historical_inputs_after.json','smoke_pair.json',*[f'matched_seed{s}.json' for s in SEEDS]]:
  if available(sftp,'evidence/'+name):download(sftp,'evidence/'+name,folder/'evidence'/name)
 return folder

def publish(folder,status):
 if not clean_checkout():return {'status':'COLLECTED_REPOSITORY_HAS_OTHER_CHANGES','folder':str(folder)}
 assert git('remote','get-url','origin')=='https://github.com/Polaris1010/EM-DINOv3-Research.git'
 api=verify_private_remote()
 original=json.loads((ROOT/'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:32]
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 snapshot=ROOT/'evidence/MV-3/v2'/f'final_status_{stamp}.json';snapshot.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8',newline='\n')
 target=ROOT/'experiments/MV-3/v2'
 import shutil
 for path in folder.rglob('*'):
  if not path.is_file():continue
  relative=path.relative_to(folder);dst=(ROOT/'docs/MV3_final_report.md') if relative.as_posix()=='MV3_final_report.md' else target/relative
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
 index_path=target/'run_index.json';index=json.loads(index_path.read_text(encoding='utf-8'));index['observed_at']=stamp;index['status_evidence']=snapshot.relative_to(ROOT).as_posix()
 for run in index['runs']:
  completion=target/'runs'/run['model']/f"seed{run['seed']}"/'training_complete.json'
  if completion.exists():
   j=json.loads(completion.read_text());run.update(completed_steps=j['completed_steps'],decoder_initial_sha256=j['decoder_initial_sha256'],sampling_hash=j['sampling_hash'])
  if status['status']=='complete':
   side=json.loads((completion.parent/'head_step040000.json').read_text());assert side['step']==40000
   run.update(status='VERIFIED_COMPLETED',execution_state='TRAIN_AND_FULL_DEVELOPMENT_COMPLETE',checkpoint_path=side['path'],checkpoint_sha256=side['sha256'],metrics_path='experiments/MV-3/v2/MV3_per_region_metrics.csv',report_path='docs/MV3_final_report.md')
  else:run['execution_state']='PIPELINE_STOPPED_'+status.get('phase','UNKNOWN')
 index_path.write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
 logfile=ROOT/'docs/work_logs/20261014_05_mv3_v2_terminal_collection.md'
 logfile.write_text('# MV3自动结果收取\n\n实际终止状态：'+status['status']+'；UTC快照 '+stamp+'。\n\n'+('6个run各40000 steps、228条开发指标、18个checkpoint均完成。MV3_final_report.md、指标/配对/曲线/哈希已收取；任务结束，不启动新训练。' if status['status']=='complete' else '执行器停止；仅保存实际产物，不把缺失指标或未完成run写成完成。查看STATUS.json和服务器logs。')+'\n\n原26条历史与6条MV3 v1 registry快照保持不变。自动收取不访问图像、不启动实验、不修改网站/环境。\n',encoding='utf-8',newline='\n')
 logindex=ROOT/'docs/work_logs/README.md';text=logindex.read_text(encoding='utf-8');text+='\n- [MV3最终状态与自动收取](20261014_05_mv3_v2_terminal_collection.md)\n';logindex.write_text(text,encoding='utf-8',newline='\n')
 state=ROOT/'docs/PROJECT_STATE.md';text=state.read_text(encoding='utf-8');state.write_text(text.replace('# 当前研究状态','# 当前研究状态\n\n最新终止状态 '+stamp+'：MV3 '+status['status']+'；实际产物见experiments/MV-3/v2，完整完成时报告为docs/MV3_final_report.md。',1),encoding='utf-8',newline='\n')
 for name in ['NEW_WINDOW_HANDOFF.md','NEXT_ACTION.md']:
  p=ROOT/'docs'/name;text=p.read_text(encoding='utf-8')
  notice='> 最新MV3终止状态 '+stamp+'：'+status['status']+'。所有6组完成，先读MV3_final_report.md；不重启，不自动启动新实验。' if status['status']=='complete' else '> MV3执行器已停止，结果未全部完成；先读experiments/MV-3/v2/STATUS.json，禁止声称完整完成。'
  p.write_text(notice+'\n\n'+text,encoding='utf-8',newline='\n')
 for args in [['scripts/update_registry.py'],['scripts/update_registry.py','--check'],['scripts/validate_repository.py'],['-m','unittest','discover','-s','tests','-v']]:
  p=subprocess.run([sys.executable,*args],cwd=ROOT,env=ENV,text=True,capture_output=True)
  if p.returncode:raise RuntimeError(p.stdout[-2000:]+p.stderr[-2000:])
 assert json.loads((ROOT/'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:32]==original
 git('diff','--check');git('add','.');git('commit','-q','-m','Collect actual Han full-GT MV3 v2 results and final research handoff');commit=git('rev-parse','HEAD')
 for attempt in range(12):
  try:
   git('push','origin',BRANCH)
   body=(ROOT/'docs/MV3_PR.md').read_text(encoding='utf-8')+'\n\nActual terminal state: '+status['status']+'; collected commit '+commit+'.\n'
   if status['status']=='complete':body+='[MV3 final report](https://github.com/Polaris1010/EM-DINOv3-Research/blob/'+commit+'/docs/MV3_final_report.md). All six runs, 18 checkpoint hashes and 228 development metric rows are collected.\n'
   api('/repos/Polaris1010/EM-DINOv3-Research/pulls/2','PATCH',{'body':body})
   return {'status':'PUSHED','commit':commit,'folder':str(folder),'pipeline_status':status['status'],'pr':'https://github.com/Polaris1010/EM-DINOv3-Research/pull/2'}
  except RuntimeError:
   if attempt==11:raise
   time.sleep(60)

def watch(password,max_seconds=48*3600,poll_seconds=120,retry_seconds=30):
 client=None;deadline=time.time()+max_seconds
 try:
  while time.time()<deadline:
   terminal=None
   try:
    if client is None:
     client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
     client.connect('192.168.203.97',username='root',password=password,look_for_keys=False,allow_agent=False,timeout=min(60,max(1,deadline-time.time())))
     client.get_transport().set_keepalive(30)
    with client.open_sftp() as sftp:
     status=read_remote(sftp,'STATUS.json')
     print(datetime.datetime.now(datetime.timezone.utc).isoformat(),status.get('status'),status.get('phase'),flush=True)
     if status.get('status') in ['complete','failed']:terminal=collect(sftp,status)
   except paramiko.AuthenticationException:raise
   except (OSError,EOFError,paramiko.SSHException) as error:
    if client is not None:client.close()
    client=None
    print('SSH_RECONNECT_PENDING',type(error).__name__,flush=True)
    time.sleep(min(retry_seconds,max(0,deadline-time.time())));continue
   if terminal is not None:
    # Publishing errors are kept separate from transport retries; do not repeat Git writes.
    result=publish(terminal,status);(SCRATCH/'collector_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result),flush=True);return
   time.sleep(min(poll_seconds,max(0,deadline-time.time())))
  raise TimeoutError('Bounded collection window expired; server outputs preserved')
 finally:
  if client is not None:client.close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--watch-stdin',action='store_true');ap.add_argument('--launch',action='store_true');a=ap.parse_args()
 if a.launch:
  password=getpass.getpass('SSH password: ');log=(SCRATCH/'collector.log').open('a',encoding='utf-8')
  p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch-stdin'],stdin=subprocess.PIPE,stdout=log,stderr=log,text=True,env=ENV,creationflags=subprocess.CREATE_NO_WINDOW)
  p.stdin.write(password+'\n');p.stdin.close();log.close();(SCRATCH/'collector_pid.json').write_text(json.dumps({'pid':p.pid,'role':'finite_passive_result_collector','server_root':REMOTE,'credentials_saved':False})+'\n',encoding='utf-8',newline='\n');print('PASSIVE_COLLECTOR_STARTED',p.pid)
 else:
  try:watch(sys.stdin.readline().rstrip('\r\n') if a.watch_stdin else getpass.getpass('SSH password: '))
  except Exception as error:
   (SCRATCH/'collector_result.json').write_text(json.dumps({'status':'COLLECTION_NEEDS_ATTENTION','error':type(error).__name__+': '+str(error),'collected_folder':str(SCRATCH/'collected_v2_results')})+'\n',encoding='utf-8',newline='\n');raise
