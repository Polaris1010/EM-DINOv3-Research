"""Single coordinator resource guard, adapted from round3; only owns registered process groups."""
import os,sys,time,json,csv,subprocess,signal,threading,collections,urllib.request
from pathlib import Path
import psutil
BASE=Path(__file__).resolve().parent
SERVICE='/opt/data/3_117/tanghr/mira_superhuman_demo_20260825'
GIB=1024**3

def run(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=40)
def gpu_state():
 rows=csv.reader(run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits']).splitlines())
 return {int(r[0]):dict(uuid=r[1].strip(),total=int(r[2]),used=int(r[3]),free=int(r[4]),util=int(r[5]),temp=int(r[6])) for r in rows}
def apps(g):
 rows=csv.reader(run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']).splitlines());result=[]
 for r in rows:
  if len(r)<3:continue
  gpu=next((i for i,x in g.items() if x['uuid']==r[0].strip()),None)
  if gpu not in [1,2,3]:continue
  try:
   p=psutil.Process(int(r[1]))
   if not p.is_running() or p.status()==psutil.STATUS_ZOMBIE:continue
   result.append((gpu,p,int(r[2])))
  except psutil.NoSuchProcess:continue
 return result
def website(p):
 try:return p.username()=='root' and p.cwd()==SERVICE and p.cmdline()[0]==SERVICE+'/venv/bin/python' and 'uvicorn' in p.cmdline() and 'server.app:app' in p.cmdline()
 except psutil.Error:return False

def trend_stop(hist):
 if len(hist)<20 or hist[-1][0]-hist[0][0]<570:return None
 first,last=hist[0],hist[-1];rss_growth=last[1]-first[1];swap_growth=last[2]-first[2]
 # Four successive 2.5-minute window means, no plateau.
 means=[sum(x[1] for x in list(hist)[i:i+5])/5 for i in [0,5,10,15]]
 if rss_growth>12*GIB and all(b-a>.25*GIB for a,b in zip(means,means[1:])):return 'RSS grew >12 GiB in 10-minute window without plateau'
 if swap_growth>GIB and (rss_growth>.5*GIB or first[3]-last[3]>GIB):return 'Swap grew >1 GiB with own RSS growth or falling MemAvailable'
 return None

class Guard:
 def __init__(self):
  self.lock=threading.RLock();self.jobs={};self.abort=threading.Event();self.reason=None;self.stop_monitor=threading.Event();self.history=collections.deque(maxlen=21);self.site_fail=0
  self.web={};self.reslog=(BASE/'resource_log.csv').open('a',newline='');self.writer=csv.DictWriter(self.reslog,fieldnames=['timestamp','gpu_state','gpu_processes','rss_tree_gib','mem_available_gib','mem_total_gib','swap_gib','iowait_percent','website_http'])
  if self.reslog.tell()==0:self.writer.writeheader()
 def owned(self,p):
  with self.lock:
   for j in self.jobs.values():
    try:
     if p.pid==j['p'].pid and abs(p.create_time()-j['birth'])<.1 and p.cmdline()==j['cmd'] and p.cwd()==j['cwd']:return True
    except psutil.Error:continue
  return False
 def snapshot(self,baseline=False):
  g=gpu_state();aa=apps(g);processes=[]
  for gpu,p,mem in aa:
   if website(p):
    key=f'{gpu}:{p.pid}'
    if baseline:self.web[key]={'memory':mem,'birth':p.create_time()}
    elif key not in self.web or abs(p.create_time()-self.web[key]['birth'])>.1:raise RuntimeError('Website identity changed')
    elif mem-self.web[key]['memory']>4096 or g[gpu]['free']<2048:raise RuntimeError('Website growth exceeded 4 GiB or shared GPU reserve below 2 GiB')
   elif not self.owned(p):
    try:
     if not p.is_running() or p.status()==psutil.STATUS_ZOMBIE:continue
    except psutil.NoSuchProcess:continue
    raise RuntimeError(f'Unknown GPU process on {gpu}: {p.pid}')
   processes.append([gpu,p.pid,mem])
  present={f'{gpu}:{p.pid}' for gpu,p,mem in aa if website(p)}
  if not baseline and set(self.web)-present:raise RuntimeError('Website CUDA process disappeared')
  try:
   with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=5) as r:code=r.status;r.read(64)
   if code!=200:raise RuntimeError(code)
   self.site_fail=0
  except Exception:
   code='failed';self.site_fail+=1
   if self.site_fail>=2 or baseline:raise RuntimeError('Website HTTP health failed')
  rss=0
  with self.lock:
   for j in self.jobs.values():
    if j['p'].poll() is not None:continue
    try:
     p=psutil.Process(j['p'].pid)
     for q in [p]+p.children(recursive=True):
      try:rss+=q.memory_info().rss
      except psutil.Error:pass
    except psutil.Error:pass
  m=psutil.virtual_memory();sw=psutil.swap_memory();io=getattr(psutil.cpu_times_percent(),'iowait',0.);now=time.time()
  self.history.append((now,rss,sw.used,m.available))
  self.writer.writerow(dict(timestamp=now,gpu_state=json.dumps(g),gpu_processes=json.dumps(processes),rss_tree_gib=rss/GIB,mem_available_gib=m.available/GIB,mem_total_gib=m.total/GIB,swap_gib=sw.used/GIB,iowait_percent=io,website_http=code));self.reslog.flush()
  if rss>96*GIB:raise RuntimeError('Combined job process-tree RSS >96 GiB')
  if m.available<max(64*GIB,.15*m.total):raise RuntimeError('MemAvailable threshold')
  if any(g[i]['temp']>=85 for i in [1,2,3]):raise RuntimeError('GPU temperature >=85C')
  d=psutil.disk_usage(BASE);st=os.statvfs(BASE)
  if d.free/d.total<.15 or st.f_favail/st.f_files<.10:raise RuntimeError('Disk/inode reserve below15%')
  reason=trend_stop(self.history)
  if reason:raise RuntimeError(reason)
  return g,aa
 def stop_all(self,reason):
  with self.lock:
   if self.reason is None:self.reason=reason
   self.abort.set();(BASE/'STOP.json').write_text(json.dumps({'reason':self.reason,'time':time.time()}))
   jobs=list(self.jobs.values())
  for j in jobs:
   p=j['p']
   if p.poll() is None:
    try:
     q=psutil.Process(p.pid)
     if abs(q.create_time()-j['birth'])<.1 and q.cmdline()==j['cmd'] and q.cwd()==j['cwd'] and os.getpgid(p.pid)==p.pid:os.kill(p.pid,signal.SIGUSR1)
    except (psutil.Error,ProcessLookupError):pass
  deadline=time.time()+120
  for j in jobs:
   try:j['p'].wait(timeout=max(.1,deadline-time.time()))
   except subprocess.TimeoutExpired:
    try:
     q=psutil.Process(j['p'].pid)
     if abs(q.create_time()-j['birth'])<.1 and os.getpgid(q.pid)==q.pid:os.killpg(q.pid,signal.SIGKILL)
    except (psutil.Error,ProcessLookupError):pass
 def monitor(self):
  while not self.stop_monitor.wait(30):
   try:self.snapshot()
   except Exception as e:self.stop_all(repr(e));return
 def start(self):
  with (BASE/'gpu_preflight.txt').open('w') as f:
   for cmd in [['nvidia-smi'],['free','-h'],['vmstat','5','7']]:f.write(' '.join(cmd)+'\n'+run(cmd)+'\n');f.flush()
  self.snapshot(baseline=True);(BASE/'allowed_gpu_processes.json').write_text(json.dumps(self.web,indent=2));self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
 def job(self,name,gpu,script,args,python=None):
  assert gpu in [1,2,3] or gpu is None
  out=BASE/'jobs'/name;out.mkdir(parents=True,exist_ok=True)
  if (out/'guard_result.json').exists():
   old=json.loads((out/'guard_result.json').read_text())
   if old['exit_code']==0:return out
   raise RuntimeError('Prior failed run requires explicit revised run name')
  if self.abort.is_set():raise RuntimeError(self.reason)
  if gpu is not None:
   deadline=time.time()+120
   while True:
    g=gpu_state();aa=apps(g)
    if not any(i==gpu and not website(p) for i,p,mem in aa) and g[gpu]['used']<=sum(mem for i,p,mem in aa if i==gpu and website(p))+256 and g[gpu]['free']>18*1024 and g[gpu]['util']<=5:break
    if time.time()>deadline:raise RuntimeError(f'GPU {gpu} failed idle release')
    if self.abort.wait(5):raise RuntimeError(self.reason)
  cmd=[python or sys.executable,'-u',str(BASE/script)]+list(map(str,args));env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',RUN_OUTPUT=str(out),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',nnUNet_compile='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
  log=(out/'train.log').open('w')
  with self.lock:
   if self.abort.is_set():raise RuntimeError(self.reason)
   p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=out,env=env,start_new_session=True)
   j=dict(p=p,birth=psutil.Process(p.pid).create_time(),cmd=cmd,cwd=str(out));self.jobs[name]=j
   registry={k:dict(pid=v['p'].pid,pgid=v['p'].pid,create_time=v['birth'],command=v['cmd'],cwd=v['cwd']) for k,v in self.jobs.items()}
   (BASE/'process_registry.json').write_text(json.dumps(registry,indent=2))
   (out/'pid.json').write_text(json.dumps(dict(pid=p.pid,create_time=j['birth'],command=cmd,cwd=str(out),gpu=gpu)))
  code=p.wait();log.close();(out/'guard_result.json').write_text(json.dumps({'exit_code':code,'stop_reason':self.reason,'gpu':gpu}))
  if code!=0:self.stop_all(f'{name} exited {code}');raise RuntimeError(f'{name} failed')
  return out
 def close(self):self.stop_monitor.set();self.thread.join(timeout=40);self.reslog.close()
