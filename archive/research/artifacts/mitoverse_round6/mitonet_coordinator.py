"""Transfer the one live guard, retain seed2, and fill GPU1/3 with MitoNet."""
import os, json, time, signal, subprocess, threading, traceback, hashlib, concurrent.futures
from pathlib import Path
import psutil
from settings import *
from guard import Guard,save

class Adopted:
    def __init__(self,pid): self.pid=pid
    def wait(self,timeout=None):
        try:return psutil.Process(self.pid).wait(timeout)
        except psutil.NoSuchProcess:return 0
        except psutil.TimeoutExpired:raise subprocess.TimeoutExpired(str(self.pid),timeout)

def report():
    import runpy
    runpy.run_path(str(BASE/'mitonet_report.py'),run_name='__main__')

def main():
    assert not (BASE/'mitonet_handover_committed.json').exists()
    prior=json.loads((BASE/'coordinator_pid.json').read_text());old=psutil.Process(prior['pid'])
    assert abs(old.create_time()-prior['birth'])<.1
    assert old.cmdline()[-1]==str(BASE/'start_exploratory_baseline.py')
    assert not (BASE/'STOP.json').exists()
    # All preparation precedes the brief, rollback-protected guard handover.
    g=Guard();started=False;committed=False
    save(BASE/'coordinator_before_mitonet.json',prior)
    try:
        old.send_signal(signal.SIGSTOP)
        for _ in range(100):
            if old.status()==psutil.STATUS_STOPPED:break
            time.sleep(.01)
        assert old.status()==psutil.STATUS_STOPPED
        registry=json.loads((BASE/'process_registry.json').read_text());save(BASE/'registry_before_mitonet.json',registry)
        for name,j in registry.items():
            try:
                p=psutil.Process(j['pid'])
                if p.status()==psutil.STATUS_ZOMBIE:continue
                assert abs(p.create_time()-j['birth'])<.1 and p.cmdline()==j['cmd'] and p.cwd()==j['cwd'] and os.getpgid(p.pid)==p.pid
                g.jobs[name]=dict(p=Adopted(p.pid),birth=j['birth'],cmd=j['cmd'],cwd=j['cwd'])
            except psutil.NoSuchProcess:pass
        assert list(g.jobs)==['masked_P1_seed2_10000'], 'Unexpected active jobs; resume old guard'
        for line in (BASE/'resources.jsonl').open():
            row=json.loads(line)
            if row['time']>=prior['started']:
                g.swap0=g.swap0._replace(used=row['swap_used']);break
        g.kernel_errors=set(json.loads((BASE/'hardware_error_baseline.json').read_text()))
        g.start();started=True
        save(BASE/'coordinator_pid.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),command=['mitonet_coordinator.py'],started=time.time(),adopted_from=prior))
        save(BASE/'mitonet_handover_committed.json',dict(old_pid=old.pid,new_pid=os.getpid(),time=time.time(),adopted=list(g.jobs),heartbeat='same path; no worker interruption',website_baseline='preserved'))
        committed=True
        old.kill()  # Only the verified old coordinator; never its worker process groups.
        print('GUARD_HANDOVER_OK',flush=True)
    except Exception:
        if not committed:
            try:old.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:pass
        if started:g.close()
        raise
    def mito(shard,gpu):
        g.job(f'mitonet_M0_shard{shard}_smoke',gpu,'mitonet_compare.py',['--shard',str(shard),'--smoke'])
        g.job(f'mitonet_M0_shard{shard}_full',gpu,'mitonet_compare.py',['--shard',str(shard)])
    def existing_seed():
        name='masked_P1_seed2_10000';j=g.jobs[name];p=Path(j['cwd'])
        while g.own_identity(j):
            if g.abort.wait(5):raise RuntimeError(g.reason)
        assert not g.abort.is_set()
        d=json.loads((p/'result.json').read_text());assert d['status']=='complete' and d['steps']==10000
        save(p/'adopted_guard_result.json',dict(status='complete',resource_peaks=g.peaks))
        g.job('masked_P1_seed2_validation',2,'exploratory_downstream.py',['--seed','2','--evaluate-from',str(p),'--validation-only'])
        s=json.loads((BASE/'stage_status.json').read_text());s['exploratory_masked_baseline']=dict(status='completed',seeds=[0,1,2],steps_per_seed=10000,added_SSL=False);save(BASE/'stage_status.json',s)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(existing_seed),pool.submit(mito,0,1),pool.submit(mito,1,3)]
            while not all(f.done() for f in futures):
                for f in futures:
                    if f.done():f.result()
                report()
                if g.abort.wait(10):raise RuntimeError(g.reason)
            for f in futures:f.result()
        report();save(BASE/'mitonet_coordinator_result.json',dict(status='complete',resource_peaks=g.peaks,time=time.time()))
    except Exception as e:
        g.stop_all(repr(e));save(BASE/'mitonet_coordinator_error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:g.close()

if __name__=='__main__':main()
