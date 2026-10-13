"""Recover only unfinished downstream jobs; preserve completed SSL and evaluations."""
import os,json,time,queue,concurrent.futures,traceback,psutil
from settings import *
from guard import Guard,save
def complete(p):
 try:return json.loads((p/'result.json').read_text()).get('status')=='complete'
 except (OSError,ValueError):return False
def main():
 prior=json.loads((BASE/'coordinator_pid.json').read_text())
 try:
  p=psutil.Process(prior['pid'])
  assert abs(p.create_time()-prior['birth'])>.1 or p.status()==psutil.STATUS_ZOMBIE, 'previous coordinator still alive'
 except psutil.NoSuchProcess:pass
 reg=json.loads((BASE/'process_registry.json').read_text())
 for j in reg.values():
  try:
   p=psutil.Process(j['pid'])
   assert abs(p.create_time()-j['birth'])>.1 or p.status()==psutil.STATUS_ZOMBIE, 'previous worker still alive'
  except psutil.NoSuchProcess:pass
 g=Guard();g.start()
 stop=BASE/'STOP.json'
 if stop.exists():stop.rename(BASE/f'STOP_exit_race_{int(time.time())}.json')
 save(BASE/'coordinator_pid.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),started=time.time(),command=[__file__]))
 save(BASE/'exploratory_ssl_status.json',dict(status='running',phase='remaining downstream probes and full evaluations',SSL_complete=['TJ','TM'],recovery='finished worker exit race',time=time.time()))
 tasks=queue.Queue()
 for arm,seed in [('TM',0),('TM',1),('TM',2),('TJ',2),('TJ',0),('TJ',1)]:tasks.put((arm,seed))
 def job(name,gpu,args):
  p=BASE/'jobs'/name
  if complete(p):return p
  if p.exists():
   p=BASE/'jobs'/(name+'_recovery1')
   if complete(p):return p
   assert not p.exists()
  return g.job(p.name,gpu,'ssl_probe.py',args)
 def work(gpu):
  while not g.abort.is_set():
   try:arm,seed=tasks.get_nowait()
   except queue.Empty:return
   args=['--encoder',str(BASE/'jobs'/f'{arm}_b16_long10000'/'teacher_step10000.pth'),'--seed',str(seed)]
   p=job(f'{arm}_head{seed}_10000',gpu,args+['--steps','10000'])
   job(f'{arm}_head{seed}_validation',gpu,args+['--evaluate-from',str(p),'--validation-only'])
   job(f'{arm}_head{seed}_legacy',gpu,args+['--evaluate-from',str(p),'--legacy-only'])
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   fs=[pool.submit(work,i) for i in [1,2,3]]
   for f in concurrent.futures.as_completed(fs):
    try:f.result()
    except Exception as e:g.stop_all(repr(e));raise
  save(BASE/'exploratory_ssl_status.json',dict(status='complete',arms=['TJ','TM'],SSL_steps=10000,head_seeds=[0,1,2],Gram_weight=0,exploratory=True,time=time.time()))
 except Exception as e:
  g.stop_all(repr(e));save(BASE/'exploratory_ssl_status.json',dict(status='blocked',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
 finally:g.close()
if __name__=='__main__':main()
