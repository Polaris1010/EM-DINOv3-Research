"""Checkpoint the one active LT worker, then restart under the shared scheduler."""
import json,os,signal,subprocess,time,hashlib,shutil
from pathlib import Path
import psutil
S=Path(__file__).resolve().parent;R=S.parent
def run(*args,check=True):return subprocess.run(args,check=check,text=True,capture_output=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 if (S/'STOP.json').exists():raise RuntimeError('Unexpected preexisting STOP')
 reg=json.loads((S/'process_registry.json').read_text())
 candidates=[]
 for name,row in reg.items():
  if name.startswith('formal_seed0_LT_part'):
   try:
    p=psutil.Process(row['pid'])
    if abs(p.create_time()-row['birth'])<.1 and p.cmdline()==row['cmd'] and p.cwd()==row['cwd']:candidates.append((name,row,p))
   except psutil.NoSuchProcess:pass
 if len(candidates)!=1:raise RuntimeError(f'Expected one LT worker; found {len(candidates)}')
 name,row,p=candidates[0];job=Path(row['cwd'])
 assert job.parent==S/'jobs'
 evidence=R/'evidence'/f'overlap_handoff_{int(time.time())}';evidence.mkdir()
 for filename in ('formal_seed0_status.json','process_registry.json'):
  shutil.copy2(S/filename,evidence/filename)
 run('systemctl','stop','mito-pretrain-v2-watcher')
 p.send_signal(signal.SIGUSR1)
 print('Requested checkpoint at next LT step',p.pid,flush=True)
 deadline=time.time()+180
 while p.is_running() and p.status()!=psutil.STATUS_ZOMBIE:
  if time.time()>deadline:raise RuntimeError('LT checkpoint handoff timed out; inspect without forced kill')
  time.sleep(2)
 result=json.loads((job/'result.json').read_text())
 assert result['status'] in ('interrupted','complete')
 receipt=json.loads((job/'latest_checkpoint.json').read_text())
 assert receipt['step']==result['completed_steps']
 assert sha(Path(receipt['path']))==receipt['sha256']
 print('Verified saved checkpoint',receipt['step'],flush=True)
 run('systemctl','stop','mito-pretrain-v2-seed0')
 stop=S/'STOP.json'
 if stop.exists():
  reason=json.loads(stop.read_text())['reason']
  if reason!=f'{name} exited 75':raise RuntimeError(f'Unexpected STOP reason: {reason}')
  stop.rename(evidence/'intentional_handoff_STOP.json')
 (evidence/'checkpoint_receipt.json').write_text(json.dumps(receipt,indent=2))
 run('systemctl','reset-failed','mito-pretrain-v2-seed0',check=False)
 for _ in range(10):
  state=run('systemctl','show','mito-pretrain-v2-seed0','-p','LoadState','--value',check=False).stdout.strip()
  if state=='not-found':break
  time.sleep(1)
 cmd=f'exec /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u {S}/coordinator_overlap.py >> {S}/overlap_coordinator.log 2>&1'
 run('systemd-run','--unit=mito-pretrain-v2-seed0',f'--property=WorkingDirectory={S}','--property=KillMode=control-group','--property=Restart=no','/bin/bash','-c',cmd)
 time.sleep(8)
 assert run('systemctl','is-active','mito-pretrain-v2-seed0').stdout.strip()=='active'
 assert not stop.exists()
 run('systemctl','reset-failed','mito-pretrain-v2-watcher',check=False)
 cmd=f'exec /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u {S}/watch_all_batches.py >> {S}/watcher.log 2>&1'
 run('systemd-run','--unit=mito-pretrain-v2-watcher',f'--property=WorkingDirectory={S}','--property=Restart=no','/bin/bash','-c',cmd)
 print('Overlap scheduler and dependency watcher started',flush=True)

if __name__=='__main__':main()
