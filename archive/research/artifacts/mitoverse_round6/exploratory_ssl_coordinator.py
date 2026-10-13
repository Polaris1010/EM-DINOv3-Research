"""One guarded exploratory TJ/TM pair, then matched three-seed probes."""
import os,json,time,signal,psutil,hashlib,queue,threading,concurrent.futures,traceback
from settings import *
from guard import Guard,save
from mitonet_coordinator import Adopted

def main():
    assert not (BASE/'STOP.json').exists()
    prior=json.loads((BASE/'coordinator_pid.json').read_text());old=psutil.Process(prior['pid'])
    assert abs(old.create_time()-prior['birth'])<.1 and old.cmdline()[-1]==str(BASE/'recover_seed2_legacy.py')
    g=Guard();started=False;committed=False
    try:
        old.send_signal(signal.SIGSTOP)
        for _ in range(100):
            if old.status()==psutil.STATUS_STOPPED:break
            time.sleep(.01)
        assert old.status()==psutil.STATUS_STOPPED
        reg=json.loads((BASE/'process_registry.json').read_text());save(BASE/'registry_before_ssl.json',reg)
        for name,j in reg.items():
            try:
                p=psutil.Process(j['pid'])
                if p.status()==psutil.STATUS_ZOMBIE:continue
                assert abs(p.create_time()-j['birth'])<.1 and p.cmdline()==j['cmd'] and p.cwd()==j['cwd']
                g.jobs[name]=dict(p=Adopted(p.pid),birth=j['birth'],cmd=j['cmd'],cwd=j['cwd'])
            except psutil.NoSuchProcess:pass
        assert list(g.jobs)==['masked_P1_seed2_legacy']
        for line in (BASE/'resources.jsonl').open():
            r=json.loads(line)
            if r['time']>=prior['started']:g.swap0=g.swap0._replace(used=r['swap_used']);break
        g.kernel_errors=set(json.loads((BASE/'hardware_error_baseline.json').read_text()));g.start();started=True
        save(BASE/'coordinator_before_ssl.json',prior)
        save(BASE/'coordinator_pid.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),command=['exploratory_ssl_coordinator.py'],started=time.time()))
        save(BASE/'ssl_guard_handover.json',dict(status='committed',previous=prior,active=list(g.jobs),heartbeat='preserved',time=time.time()))
        committed=True;old.kill()
    except Exception:
        if not committed:
            try:old.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:pass
        if started:g.close()
        raise
    def job(name,gpu,script,args):
        try:return g.job(name,gpu,script,args)
        except Exception as e:g.stop_all(repr(e));raise
    def status(**kw):save(BASE/'exploratory_ssl_status.json',dict(time=time.time(),**kw))
    def smoke(arm,gpu):
        args=['--arm',arm,'--weight','0','--batch','16']
        # 10-step checkpoint plus one continuous reference step.
        p=job(f'{arm}_b16_smoke11',gpu,'ssl_train_exploratory.py',args+['--steps','11'])
        q=job(f'{arm}_b16_resume11',gpu,'ssl_train_exploratory.py',args+['--steps','11','--resume',str(p/'checkpoint_step10.pth')])
        job(f'{arm}_b16_resume_compare',None,'compare_ssl_resume.py',[str(p/'checkpoint_step11.pth'),str(q/'checkpoint_step11.pth')])
    tasks=queue.Queue();done=set();lock=threading.Lock()
    def probes(gpu):
        while not g.abort.is_set():
            try:arm,seed,encoder=tasks.get(timeout=2)
            except queue.Empty:
                with lock:
                    if len(done)==2:return
                continue
            args=['--encoder',str(encoder),'--seed',str(seed)]
            p=job(f'{arm}_head{seed}_10000',gpu,'ssl_probe.py',args+['--steps','10000'])
            job(f'{arm}_head{seed}_validation',gpu,'ssl_probe.py',args+['--evaluate-from',str(p),'--validation-only'])
            job(f'{arm}_head{seed}_legacy',gpu,'ssl_probe.py',args+['--evaluate-from',str(p),'--legacy-only'])
            tasks.task_done()
    def arm_run(arm,gpu):
        args=['--arm',arm,'--weight','0','--batch','16']
        p=job(f'{arm}_b16_prefix500',gpu,'ssl_train_exploratory.py',args+['--steps','500'])
        p=job(f'{arm}_b16_long10000',gpu,'ssl_train_exploratory.py',args+['--steps','10000','--resume',str(p/'checkpoint_step500.pth')])
        for seed in range(3):tasks.put((arm,seed,p/'teacher_step10000.pth'))
        with lock:done.add(arm)
        probes(gpu)
    def adopted():
        j=g.jobs['masked_P1_seed2_legacy']
        while g.own_identity(j):
            if g.abort.wait(5):raise RuntimeError(g.reason)
        assert json.loads((Path(j['cwd'])/'result.json').read_text())['status']=='complete'
        save(Path(j['cwd'])/'adopted_ssl_guard_result.json',dict(status='complete',resource_peaks=g.peaks))
        import runpy
        runpy.run_path(str(BASE/'mitonet_report_final.py'),run_name='__main__')
        save(BASE/'matched_legacy_queue.json',dict(status='complete',seeds=[0,1,2]))
        probes(3)
    try:
        status(status='running',phase='CPU configuration and deterministic data checks',Gram_weight=0,exploratory=True)
        for arm in ['TJ','TM']:
            job(f'{arm}_cpu_data_check',None,'ssl_train_exploratory.py',['--arm',arm,'--weight','0','--batch','16','--steps','11','--cpu'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(smoke,arm,gpu) for arm,gpu in [('TJ',1),('TM',2)]]
            for f in concurrent.futures.as_completed(fs):f.result()
        initial=[json.loads((BASE/'jobs'/f'{arm}_b16_smoke11'/'ssl_initialization.json').read_text())['heads_sha256'] for arm in ['TJ','TM']]
        assert initial[0]==initial[1], 'SSL heads must share initialization'
        save(BASE/'ssl_resume_gate.json',dict(status='passed',exact_resume=True,initial_heads_equal=True,heads_sha256=initial[0],batch=16))
        status(status='running',phase='500-step prefixes then full-state resume to 10k; probes queued',batch=16,Gram_weight=0,exploratory=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            fs=[pool.submit(arm_run,'TJ',1),pool.submit(arm_run,'TM',2),pool.submit(adopted)]
            for f in concurrent.futures.as_completed(fs):f.result()
        status(status='complete',arms=['TJ','TM'],SSL_steps=10000,head_seeds=[0,1,2],Gram_weight=0,exploratory=True)
    except Exception as e:
        g.stop_all(repr(e));status(status='blocked',error=repr(e),traceback=traceback.format_exc());raise
    finally:g.close()

if __name__=='__main__':main()
