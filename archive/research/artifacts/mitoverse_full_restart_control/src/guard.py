"""Round5 coordinator guard revised for the Round6 resource contract.
One instance owns all jobs, including CPU jobs. No global/user-wide kills.
"""
import os,sys,time,json,csv,subprocess,signal,threading,urllib.request
import psutil
from settings import BASE,PYTHON
GIB=1024**3
SERVICE='/opt/data/3_117/tanghr/mira_superhuman_demo_20260825'
def run(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=3)
def gpu_state():
 rows=csv.reader(run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits']).splitlines())
 return {int(r[0]):dict(uuid=r[1].strip(),total=int(r[2]),used=int(r[3]),free=int(r[4]),util=int(r[5]),temp=int(r[6])) for r in rows}
def apps(g):
 result=[]
 for r in csv.reader(run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']).splitlines()):
  if len(r)<3:continue
  gpu=next((i for i,x in g.items() if x['uuid']==r[0].strip()),None)
  if gpu not in [1,2,3]:continue
  try:
   p=psutil.Process(int(r[1]))
   if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE:result.append((gpu,p,int(r[2])))
  except psutil.NoSuchProcess:pass
 return result
def website(p):
 try:return p.username()=='root' and p.cwd()==SERVICE and p.cmdline()[0]==SERVICE+'/venv/bin/python' and 'uvicorn' in p.cmdline() and 'server.app:app' in p.cmdline()
 except psutil.Error:return False
def save(p,x):
 tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.{threading.get_ident()}.tmp');tmp.write_text(json.dumps(x,indent=2,default=str));tmp.replace(p)
class Guard:
 def __init__(self):
  self.lock=threading.RLock();self.jobs={};self.abort=threading.Event();self.finished=threading.Event();self.reason=None
  self.web=json.loads((BASE/'allowed_gpu_processes.json').read_text()) if (BASE/'allowed_gpu_processes.json').exists() else {};self.prior_baseline=bool(self.web);self.hot={};self.site_fail=0;self.last_resource=0;self.last_http=0;self.swap_active_since=None
  self.kernel_errors=None;self.swap0=psutil.swap_memory();self.swap_prev=self.swap0;self.log=(BASE/'resources.jsonl').open('a');self.heartbeat=BASE/'coordinator.heartbeat'
  self.peaks=dict(rss_bytes=0,min_gpu_free_mib={str(i):999999 for i in [1,2,3]})
 def own_identity(self,j):
  try:
   p=psutil.Process(j['p'].pid)
   return abs(p.create_time()-j['birth'])<.1 and p.cmdline()==j['cmd'] and p.cwd()==j['cwd'] and os.getpgid(p.pid)==p.pid
  except (psutil.Error,ProcessLookupError):return False
 def owned(self,p):
  with self.lock:
   for j in self.jobs.values():
    if not self.own_identity(j):continue
    if p.pid==j['p'].pid:return True
    try:
     if os.getpgid(p.pid)==j['p'].pid:return True
    except ProcessLookupError:pass
  return False
 def snapshot(self,baseline=False):
  with self.lock:return self._snapshot(baseline)
 def _snapshot(self,baseline=False):
  now=time.time();g=gpu_state();aa=apps(g)
  if not baseline and any(g[i]['uuid']!=self.mapping[i] for i in [1,2,3]):raise RuntimeError('Physical GPU UUID mapping changed')
  if baseline:self.mapping={i:g[i]['uuid'] for i in [1,2,3]};save(BASE/'gpu_mapping.json',self.mapping)
  process_rows=[]
  for gpu,p,mem in aa:
   if website(p):
    key=f'{gpu}:{p.pid}'
    if baseline and not self.prior_baseline:self.web[key]=dict(memory=mem,birth=p.create_time(),command=p.cmdline(),cwd=p.cwd(),user=p.username())
    elif key not in self.web or abs(p.create_time()-self.web[key]['birth'])>.1:raise RuntimeError('Website identity changed')
    elif mem-self.web[key]['memory']>4096:raise RuntimeError('Website GPU memory grew >4 GiB')
   elif not self.owned(p):
    # NVIDIA's process snapshot may outlive a worker that exits during identity checks.
    try:
     if not p.is_running() or p.status()==psutil.STATUS_ZOMBIE:continue
    except psutil.NoSuchProcess:continue
    raise RuntimeError(f'Unknown GPU process {p.pid} on authorized GPU {gpu}')
   process_rows.append(dict(gpu=gpu,pid=p.pid,memory_mib=mem,birth=p.create_time(),command=p.cmdline(),user=p.username()))
  if set(self.web)-{f'{i}:{p.pid}' for i,p,m in aa if website(p)}:raise RuntimeError('Website CUDA process disappeared')
  for i in [1,2,3]:
   if g[i]['free']<2048:raise RuntimeError(f'GPU {i} reserve <2 GiB')
   self.peaks['min_gpu_free_mib'][str(i)]=min(self.peaks['min_gpu_free_mib'][str(i)],g[i]['free'])
   if g[i]['temp']>80:
    self.hot.setdefault(i,now)
    if now-self.hot[i]>=30:raise RuntimeError(f'GPU {i} >80C for 30 seconds')
   else:self.hot.pop(i,None)
  if baseline or now-self.last_http>=9.5:
   self.last_http=now
   try:
    with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=2) as r:assert r.status==200;r.read(64)
    self.site_fail=0
   except Exception:self.site_fail+=1
   if self.site_fail>=3 or (baseline and self.site_fail):raise RuntimeError('Website HTTP health failure')
  # A failing nvidia-smi call fails closed; check hardware counters at resource cadence.
  if baseline or now-self.last_resource>=9.5:
   self.last_resource=now;rss=psutil.Process().memory_info().rss
   with self.lock:
    for j in self.jobs.values():
     if not self.own_identity(j):continue
     try:
      for p in [psutil.Process(j['p'].pid)]+psutil.Process(j['p'].pid).children(recursive=True):
       try:rss+=p.memory_info().rss
       except psutil.Error:pass
     except psutil.Error:pass
   m=psutil.virtual_memory();sw=psutil.swap_memory();self.peaks['rss_bytes']=max(self.peaks['rss_bytes'],rss)
   active=(sw.sin>self.swap_prev.sin or sw.sout>self.swap_prev.sout);self.swap_prev=sw
   if active:
    if self.swap_active_since is None:self.swap_active_since=now
   else:self.swap_active_since=None
   if sw.used-self.swap0.used>GIB and self.swap_active_since is not None and now-self.swap_active_since>=60:raise RuntimeError('Swap growth >1 GiB with continuous 60-second swap activity')
   if rss>96*GIB:raise RuntimeError('Round6 process-tree RSS >96 GiB')
   if m.available<max(64*GIB,.15*m.total):raise RuntimeError('MemAvailable below reserve')
   if psutil.disk_usage(BASE).free<max(50*GIB,3*7*GIB):raise RuntimeError('Disk reserve below max(50 GiB,3 next checkpoints)')
   ecc=run(['nvidia-smi','--query-gpu=index,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'])
   for row in csv.reader(ecc.splitlines()):
    if int(row[0]) in [1,2,3] and row[1].strip().isdigit() and int(row[1])>0:raise RuntimeError('Uncorrected GPU ECC error')
   hardware={line for line in run(['dmesg','--color=never']).splitlines() if ('NVRM' in line and ('Xid' in line or 'fallen off' in line)) or 'Hardware Error' in line}
   if self.kernel_errors is None:self.kernel_errors=hardware;save(BASE/'hardware_error_baseline.json',sorted(hardware))
   elif hardware-self.kernel_errors:raise RuntimeError('New kernel GPU/hardware error: '+str(sorted(hardware-self.kernel_errors)))
   pressure={p.name:p.read_text() for p in [__import__('pathlib').Path('/proc/pressure/memory')] if p.exists()}
   self.log.write(json.dumps(dict(time=now,gpu=g,processes=process_rows,rss_bytes=rss,mem_available=m.available,swap_used=sw.used,swap_in=sw.sin,swap_out=sw.sout,pressure=pressure,http_failures=self.site_fail))+'\n');self.log.flush()
  save(self.heartbeat,dict(monotonic=time.monotonic(),wall_time=time.time(),pid=os.getpid()))
  return g,aa
 def stop_all(self,reason):
  with self.lock:
   if self.abort.is_set():return
   self.reason=reason;self.abort.set();save(BASE/'STOP.json',dict(reason=reason,time=time.time()));jobs=list(self.jobs.values())
  # SIGTERM to own groups; hard release within 25 seconds, including children of exited leaders.
  groups=[]
  for j in jobs:
   if self.own_identity(j):
    groups.append(j['p'].pid)
    try:os.killpg(j['p'].pid,signal.SIGTERM)
    except ProcessLookupError:pass
  deadline=time.time()+25
  for j in jobs:
   try:j['p'].wait(timeout=max(.01,deadline-time.time()))
   except subprocess.TimeoutExpired:pass
  for pgid in groups:
   try:os.killpg(pgid,signal.SIGKILL)
   except ProcessLookupError:pass
 def monitor(self):
  target=time.monotonic()
  while not self.finished.is_set():
   target+=5
   if self.finished.wait(max(0,target-time.monotonic())):break
   try:self.snapshot()
   except Exception as e:self.stop_all(repr(e));return
 def start(self):
  self.snapshot(True);save(BASE/'allowed_gpu_processes.json',self.web)
  self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
 def job(self,name,gpu,script,args=()):
  assert gpu in [None,1,2,3]
  out=BASE/'jobs'/name;out.mkdir(parents=True,exist_ok=False)
  if self.abort.is_set():raise RuntimeError(self.reason)
  if gpu is not None:
   g,aa=self.snapshot()
   if g[gpu]['free']<18*1024 or any(i==gpu and not website(p) for i,p,m in aa):raise RuntimeError('GPU lacks clean 18 GiB startup reserve')
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else self.mapping[gpu],CUDA_DEVICE_ORDER='PCI_BUS_ID',RUN_OUTPUT=str(out),FULL_HEARTBEAT=str(self.heartbeat),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',nnUNet_compile='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR='/tmp')
  cmd=list(map(str,[('/opt/data/3_117/tanghr/dinov3_shared/bin/python' if script=='ssl_train_exploratory.py' else PYTHON),'-u',BASE/'worker.py',BASE/script,*args]));log=(out/'train.log').open('w');start=time.time()
  with self.lock:
   if self.abort.is_set():raise RuntimeError(self.reason)
   p=subprocess.Popen(cmd,cwd=out,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   j=dict(p=p,birth=psutil.Process(p.pid).create_time(),cmd=cmd,cwd=str(out));self.jobs[name]=j
   save(BASE/'process_registry.json',{k:dict(pid=v['p'].pid,birth=v['birth'],cmd=v['cmd'],cwd=v['cwd']) for k,v in self.jobs.items()})
  code=p.wait();log.close();save(out/'guard_result.json',dict(exit_code=code,gpu=gpu,seconds=time.time()-start,resource_peaks=self.peaks,stop_reason=self.reason))
  if code!=0:self.stop_all(f'{name} exited {code}');raise RuntimeError(f'{name} failed')
  if not (out/'result.json').exists():self.stop_all(f'{name} missing result.json');raise RuntimeError('Missing completion evidence')
  return out
 def close(self):
  self.finished.set();self.thread.join(timeout=10);self.log.close()
