"""Finite passive MV5 collection; no training, inference, or retry launches.

Password is read interactively into memory; never stored in arguments or files.
Only an unchanged expected checkout may be committed/pushed automatically.
"""
import argparse,datetime,getpass,json,os,subprocess,sys,time
from pathlib import Path
import paramiko
ROOT=Path(__file__).resolve().parents[1]
SCRATCH=ROOT.parent/'artifacts/mv5_control_20261015'
REMOTE='/opt/data/3_117/tanghr/codex/mv5_mitonet_adaptation_20261015'
BRANCH='codex/mv5-mitonet-adaptation'
ENV={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never','PYTHONUTF8':'1'}
def git(*args):
 p=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','-c','credential.username=Polaris1010','-c','http.sslBackend=schannel','-c','http.version=HTTP/1.1',*args],cwd=ROOT,env=ENV,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout.strip()
def remote_json(sftp,name):
 with sftp.open(REMOTE+'/'+name,'r') as f:return json.loads(f.read().decode())
def collect(sftp,status):
 target=SCRATCH/'collected_results';target.mkdir(parents=True,exist_ok=True)
 (target/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8',newline='\n')
 names=['MV5_final_report.md','MV5_per_region_metrics.csv','MV5_source_metrics.csv','MV5_source_deltas.csv','MV5_adaptation_paired_deltas.csv','MV5_checkpoint_hashes.csv','MV5_summary.json','MV5_failure_cases.csv','MV5_results_verification.json','MV5_protocol.json','training/training_complete.json','training/progress.json','smoke/smoke_complete.json','evidence/environment.json','evidence/historical_inputs_before.json','evidence/historical_inputs_after.json','assets/deployed_code.json']
 names += [f'training/model_step{s:06d}.json' for s in [10000,20000,40000]]
 for name in names:
  try:stat=sftp.stat(REMOTE+'/'+name)
  except OSError:continue
  assert stat.st_size<2_000_000,'Unexpectedly large record; preserve server file'
  p=target/name;p.parent.mkdir(parents=True,exist_ok=True);sftp.get(REMOTE+'/'+name,str(p))
 # Preserve a compact curve with measured values, never the large raw JSONL in Git.
 try:
  import csv,collections
  samples=[];last=collections.deque(maxlen=2000)
  with sftp.open(REMOTE+'/training/training_curve.jsonl','r') as stream:
   for line in stream:
    row=json.loads(line);last.append(row)
    if row['step']%100==0:samples.append(row)
  if samples:
   with (target/'MV5_training_curve.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
   (target/'MV5_training_tail.json').write_text(json.dumps({'last_2000_updates':list(last),'note':'training loss, not validation selection; no assertion of convergence'},indent=2)+'\n',encoding='utf-8',newline='\n')
 except OSError:pass
 return target
def publish(folder,status):
 if git('branch','--show-current')!=BRANCH or git('status','--porcelain'):
  return {'status':'COLLECTED_CHECKOUT_NOT_CLEAN_EXPECTED_BRANCH','folder':str(folder),'pipeline_status':status['status']}
 from watch_mv3_results import verify_private_remote
 api=verify_private_remote();assert git('remote','get-url','origin')=='https://github.com/Polaris1010/EM-DINOv3-Research.git'
 before=json.loads((ROOT/'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:40]
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');done=status['status']=='complete'
 if done:
  check=json.loads((folder/'MV5_results_verification.json').read_text());assert check['status']=='PASS' and check['rows']==304
  assert status['historical_inputs_unchanged'] and status['training_steps']==40000 and status['scored_regions']==38
  assert (folder/'evidence/historical_inputs_before.json').read_bytes()==(folder/'evidence/historical_inputs_after.json').read_bytes()
 dest=ROOT/'experiments/MV-5';changed=[]
 for path in folder.rglob('*'):
  if not path.is_file():continue
  relative=path.relative_to(folder)
  target=(ROOT/'docs/MV5_final_report.md') if relative.as_posix()=='MV5_final_report.md' else dest/relative
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());changed.append(target.relative_to(ROOT).as_posix())
 snapshot=ROOT/'evidence/MV-5'/f'terminal_status_{stamp}.json';snapshot.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8',newline='\n');changed.append(snapshot.relative_to(ROOT).as_posix())
 p=dest/'run_index.json';index=json.loads(p.read_text(encoding='utf-8'))
 index.update(status='VERIFIED_COMPLETED' if done else 'VERIFIED_FAILED',execution_state='TRAIN_AND_FULL_38_DEVELOPMENT_COMPLETE' if done else 'PIPELINE_STOPPED_'+status.get('phase','UNKNOWN'),observed_at_utc=stamp,status_evidence=snapshot.relative_to(ROOT).as_posix(),service_state='terminal_no_restart',completed_steps=40000 if done else status.get('train_step',index.get('completed_steps',0)))
 if done:
  checkpoint=json.loads((dest/'training/model_step040000.json').read_text());index.update(checkpoint_path=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],metrics_path='experiments/MV-5/MV5_per_region_metrics.csv',report_path='docs/MV5_final_report.md',results_verification='experiments/MV-5/MV5_results_verification.json')
 if not done and (dest/'training/progress.json').exists():index['completed_steps']=json.loads((dest/'training/progress.json').read_text())['step']
 p.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');changed.append(p.relative_to(ROOT).as_posix())
 text='MV5已完成：MitoNet-MV 40k、38区域/12来源开发指标完成；先读docs/MV5_final_report.md。任务结束，不启动新实验。' if done else 'MV5执行器停止，结果未全部完成；先读experiments/MV-5/STATUS.json和run_index，不盲目重启。'
 for name in ['AGENTS.md','START_HERE.md','docs/NEW_WINDOW_HANDOFF.md','docs/PROJECT_STATE.md','docs/NEXT_ACTION.md']:
  p=ROOT/name;p.write_text('> '+text+' UTC '+stamp+'\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');changed.append(name)
 log=ROOT/'docs/work_logs/20261015_11_mv5_mitonet_adaptation.md'
 with log.open('a',encoding='utf-8',newline='\n') as f:f.write('\n## 实际终止收取 '+stamp+'\n\n'+text+' 原40条registry保持；实际checkpoint SHA、指标、配置与本轮Markdown同步提交推送至draft PR6，不自动合并。自动收取未启动任何实验。\n')
 changed.extend([log.relative_to(ROOT).as_posix(),'experiments/registry.yaml'])
 for args in [['scripts/update_registry.py'],['scripts/update_registry.py','--check'],['scripts/validate_repository.py'],['-m','unittest','discover','-s','tests','-v']]:
  r=subprocess.run([sys.executable,*args],cwd=ROOT,env=ENV,text=True,capture_output=True)
  if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
 assert json.loads((ROOT/'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:40]==before
 git('diff','--check');git('add',*sorted(set(changed)));git('commit','-m','Collect actual MV5 terminal results and research handoff');commit=git('rev-parse','HEAD')
 for attempt in range(8):
  try:git('push','origin',BRANCH);break
  except RuntimeError:
   if attempt==7:raise
   time.sleep(30)
 pr=wait_pr_head(api,commit)
 body=pr['body'].split('\n\nActual MV5 terminal state:')[0]+'\n\nActual MV5 terminal state: '+status['status']+'; collected commit '+commit+'.'
 if done:
  summary=json.loads((folder/'MV5_summary.json').read_text());body+='\n\nMitoNet-MV source-equal Dice '+str(summary['source_macro']['MitoNet-MV']['dice'])+'; B100 minus adapted '+str(summary['B100_minus_adapted'])+'. [Final report](https://github.com/Polaris1010/EM-DINOv3-Research/blob/'+commit+'/docs/MV5_final_report.md).'
 api('/repos/Polaris1010/EM-DINOv3-Research/pulls/6','PATCH',{'body':body})
 return {'status':'PUSHED','pipeline_status':status['status'],'commit':commit,'pr':'https://github.com/Polaris1010/EM-DINOv3-Research/pull/6','further_experiments_started':False}

def wait_pr_head(api,commit):
 """Git push can finish before the PR snapshot reflects the new head."""
 for attempt in range(6):
  pr=api('/repos/Polaris1010/EM-DINOv3-Research/pulls/6')
  if pr['head']['sha']==commit:return pr
  if attempt<5:time.sleep(2)
 raise RuntimeError('Branch push completed; PR head receipt did not catch up. Preserve commit, do not publish twice.')
def watch(password,max_seconds=48*3600):
 deadline=time.monotonic()+max_seconds;client=None;last_state=None
 try:
  while time.monotonic()<deadline:
   terminal=None
   try:
    if client is None:
     client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy());client.connect('192.168.203.97',username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20);client.get_transport().set_keepalive(30)
    with client.open_sftp() as sftp:
     status=remote_json(sftp,'STATUS.json');phase=status.get('phase',status['status'])
     if phase!=last_state:print('MV5_PASSIVE',phase,flush=True);last_state=phase
     if status['status'] in ['complete','failed']:
      folder=collect(sftp,status)
      # Collection waits for the finite service to exit, without changing service state.
      _,out,_=client.exec_command('systemctl show mv5-mitonet-adaptation-20261015.service -p ActiveState -p MainPID',timeout=15)
      service=out.read().decode();(folder/'service_terminal.txt').write_text(service,encoding='utf-8',newline='\n')
      if 'MainPID=0' in service:terminal=(folder,status)
   except paramiko.AuthenticationException:raise
   except (OSError,EOFError,paramiko.SSHException) as e:
    print('MV5_PASSIVE_RECONNECT',type(e).__name__,flush=True)
    if client:client.close();client=None
   if terminal is not None:
    result=publish(*terminal);print(json.dumps(result),flush=True);return result
   time.sleep(min(30,max(0,deadline-time.monotonic())))
  return {'status':'PASSIVE_DEADLINE_NO_EXPERIMENT_ACTION','further_experiments_started':False}
 finally:
  if client:client.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--max-hours',type=float,default=48);a=p.parse_args()
 assert 0<a.max_hours<=48
 password=getpass.getpass('SSH password (memory only): ')
 (SCRATCH/'collector_pid.json').write_text(json.dumps({'pid':os.getpid(),'server_root':REMOTE,'branch':BRANCH,'max_hours':a.max_hours,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'passive collect and publish terminal result only'},indent=2)+'\n',encoding='utf-8',newline='\n')
 try:result=watch(password,a.max_hours*3600)
 except BaseException as error:
  result={'status':'PASSIVE_COLLECTOR_STOPPED_ERROR','error_type':type(error).__name__,'message':str(error)[-1000:],'further_experiments_started':False}
  (SCRATCH/'collector_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');raise
 else:(SCRATCH/'collector_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
