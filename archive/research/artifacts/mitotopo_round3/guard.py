import os,sys,time,csv,json,subprocess,signal,collections
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
SERVICE='/opt/data/3_117/tanghr/mira_superhuman_demo_20260825'
GPU=3
BASE=OUT

def run(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
def gpu_state():
    rows=list(csv.reader(run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits']).splitlines()))
    return {int(r[0]):dict(uuid=r[1].strip(),total=int(r[2]),used=int(r[3]),free=int(r[4]),util=int(r[5]),temp=int(r[6])) for r in rows}
def check_processes(g,own=None):
    pids=[]
    for r in csv.reader(run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).splitlines()):
        if len(r)<2:continue
        uuid,pid=r[0].strip(),int(r[1]);idx=next((i for i,v in g.items() if v['uuid']==uuid),None)
        if idx not in (1,2,3):continue
        pids.append((idx,pid))
        if pid==own:continue
        try:
            p=psutil.Process(pid);cmd=p.cmdline()
            allowed=p.cwd()==SERVICE and p.username()=='root' and 'uvicorn' in cmd and 'server.app:app' in cmd and cmd[0]==SERVICE+'/venv/bin/python'
        except psutil.NoSuchProcess:continue
        except psutil.Error:allowed=False
        if not allowed:
            for record in (BASE/'jobs').glob('*/pid.json'):
                try:
                    entry=json.loads(record.read_text())
                    if entry['pid']==pid and abs(p.create_time()-entry['create_time'])<.1 and p.cmdline()==entry['command'] and p.cwd()==entry['cwd'] and p.username()=='root':allowed=True;break
                except (OSError,ValueError,psutil.Error):pass
        if not allowed:
            try:
                if not p.is_running() or p.status()==psutil.STATUS_ZOMBIE:continue
            except psutil.NoSuchProcess:continue
            raise RuntimeError(f'Unknown/non-website process GPU {idx} PID {pid}')
    return pids

def main():
    global OUT,GPU
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--gpu',type=int,choices=[1,2,3],required=True);parser.add_argument('--python',default=sys.executable);parser.add_argument('job',nargs=argparse.REMAINDER);args=parser.parse_args()
    GPU=args.gpu;OUT=BASE/'jobs'/args.run;OUT.mkdir(parents=True,exist_ok=True)
    assert args.job and Path(args.job[0]).is_file()
    if (OUT/'train.log').exists():raise RuntimeError('Refuse to overwrite existing run; use a new output directory')
    commands=[['nvidia-smi'],['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv'],['nvidia-smi','pmon','-c','1'],['free','-h'],['head','-n','20','/proc/meminfo'],['df','-h',str(OUT)],['df','-ih',str(OUT)]]
    with (OUT/'gpu_preflight.txt').open('w') as f:
        for cmd in commands:f.write('$ '+' '.join(cmd)+'\n'+run(cmd)+'\n')
        swaps=[]
        # Wait for the previous own CUDA context to release; all normal preflight checks still follow.
        for attempt in range(13):
            g=gpu_state();pids=check_processes(g)
            assert not any(idx==GPU and psutil.Process(pid).cwd()!=SERVICE for idx,pid in pids),'Selected GPU already has a job'
            if g[GPU]['used']<500 and g[GPU]['free']>=22*1024 and g[GPU]['util']<=5:break
            if attempt==12:raise RuntimeError('GPU did not return to idle within release grace period')
            f.write(json.dumps({'release_wait':attempt,'gpu':g[GPU]})+'\n');f.flush();time.sleep(5)
        for i in range(7):
            g=gpu_state();pids=check_processes(g)
            m=psutil.virtual_memory();sw=psutil.swap_memory();swaps.append(sw.used)
            f.write(json.dumps({'time':time.time(),'gpus':g,'pids':pids,'mem_available':m.available,'swap_used':sw.used})+'\n');f.flush()
            assert not any(idx==GPU and psutil.Process(pid).cwd()!=SERVICE for idx,pid in pids),'Selected GPU already has a job'
            assert g[GPU]['used']<500,'Selected GPU exceeds permitted idle website footprint'
            assert g[GPU]['used']<500 and g[GPU]['free']>=22*1024 and g[GPU]['util']<=5
            assert m.available>=max(m.total*.30,64*1024**3)
            disk=psutil.disk_usage(str(OUT));assert disk.free/disk.total>=.15
            v=os.statvfs(OUT);assert v.f_favail/v.f_files>=.15
            if len(swaps)>=3 and all(b>a for a,b in zip(swaps[-3:],swaps[-2:])):raise RuntimeError('Swap increasing during preflight')
            if i<6:time.sleep(5)
        f.write('FINAL RECHECK\n'+run(['nvidia-smi']));f.flush()
        g=gpu_state();pids=check_processes(g);assert g[GPU]['used']<500 and g[GPU]['free']>=22*1024 and g[GPU]['util']<=5
    env=os.environ.copy();env.update(RUN_OUTPUT=str(OUT),CUDA_VISIBLE_DEVICES=str(GPU),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',nnUNet_compile='false')
    cmd=[args.python,'-u']+args.job
    (OUT/'command.txt').write_text('CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='+str(GPU)+' '+' '.join(cmd)+'\nGuard entry: '+sys.executable+' '+str(OUT/'guard.py'))
    log=(OUT/'train.log').open('w');child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=OUT)
    (OUT/'training_pid.txt').write_text(str(child.pid));(OUT/'pid.json').write_text(json.dumps({'pid':child.pid,'create_time':psutil.Process(child.pid).create_time(),'command':cmd,'cwd':str(OUT),'gpu':GPU}));reason=None;history=collections.deque(maxlen=130);io_bad=0
    fields=['timestamp','pid','gpu_physical_id','cuda_visible_devices','gpu_used_mib','gpu_free_mib','gpu_util_percent','gpu_temp_c','rss_gib','rss_tree_gib','mem_available_gib','mem_total_gib','swap_used_gib','iowait_percent']
    try:
        with (OUT/'resource_log.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            while child.poll() is None:
                g=gpu_state();check_processes(g,child.pid);m=psutil.virtual_memory();sw=psutil.swap_memory();p=psutil.Process(child.pid);rss=p.memory_info().rss;tree=rss
                for c in p.children(recursive=True):
                    try:tree+=c.memory_info().rss
                    except psutil.Error:pass
                io=getattr(psutil.cpu_times_percent(),'iowait',0.);now=time.time();history.append((now,rss,sw.used));io_bad=io_bad+1 if io>30 else 0
                w.writerow(dict(timestamp=now,pid=child.pid,gpu_physical_id=GPU,cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],gpu_used_mib=g[GPU]['used'],gpu_free_mib=g[GPU]['free'],gpu_util_percent=g[GPU]['util'],gpu_temp_c=g[GPU]['temp'],rss_gib=rss/1024**3,rss_tree_gib=tree/1024**3,mem_available_gib=m.available/1024**3,mem_total_gib=m.total/1024**3,swap_used_gib=sw.used/1024**3,iowait_percent=io));f.flush()
                if m.available<max(m.total*.15,64*1024**3):raise RuntimeError('System memory stop threshold')
                if g[GPU]['temp']>=85 or g[GPU]['free']<1024:raise RuntimeError('GPU temperature/memory reserve stop threshold')
                if len(history)>=4 and all(b[2]>a[2] for a,b in zip(list(history)[-4:],list(history)[-3:])):raise RuntimeError('Swap continuously growing')
                if len(history)>=120 and history[-1][0]-history[0][0]>=600 and all(b[1]>a[1] for a,b in zip(list(history),list(history)[1:])):raise RuntimeError('RSS growing for ten minutes')
                if io_bad>=12:raise RuntimeError('Sustained abnormal IO wait')
                if psutil.disk_usage(str(OUT)).percent>85:raise RuntimeError('Output disk below reserve')
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
    except Exception as e:
        reason=repr(e)
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=20)
    finally:
        log.close();(OUT/'guard_result.json').write_text(json.dumps({'exit_code':child.wait(),'stop_reason':reason,'physical_gpu':GPU,'cuda_visible_devices':str(GPU)},indent=2))
    print((OUT/'guard_result.json').read_text(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT!=BASE:(OUT/'guard_result.json').write_text(json.dumps({'exit_code':1,'stop_reason':'guard error: '+repr(e),'physical_gpu':GPU}))
        raise

