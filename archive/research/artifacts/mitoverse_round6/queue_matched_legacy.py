"""No concurrent guard: wait for current coordinator, then use GPU1/2/3 for matched tests."""
import time,json,os,concurrent.futures,psutil,runpy,hashlib
from settings import *
from guard import Guard,save

def main():
    current=json.loads((BASE/'coordinator_pid.json').read_text())
    assert current['command']==['mitonet_coordinator.py']
    save(BASE/'matched_legacy_queue.json',dict(status='waiting',pid=os.getpid(),depends_on=current,plan={'1':'P1 corrected seed0','2':'P1 corrected seed1','3':'P1 corrected seed2'},scope='one full fixed legacy-dev evaluation per seed; no tuning'))
    while psutil.pid_exists(current['pid']):
        p=psutil.Process(current['pid'])
        if abs(p.create_time()-current['birth'])>.1 or p.status()==psutil.STATUS_ZOMBIE:break
        time.sleep(10)
    assert not (BASE/'STOP.json').exists(), 'Prior guard stopped; no automatic restart'
    assert json.loads((BASE/'mitonet_coordinator_result.json').read_text())['status']=='complete'
    assert json.loads((BASE/'coordinator_pid.json').read_text())==current, 'Another coordinator took ownership'
    save(BASE/'coordinator_before_matched_legacy.json',current)
    save(BASE/'coordinator_pid.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),command=['queue_matched_legacy.py'],started=time.time()))
    save(BASE/'matched_legacy_queue.json',dict(status='running',pid=os.getpid(),seeds=[0,1,2],script_sha256=hashlib.sha256((BASE/'corrected_legacy_eval.py').read_bytes()).hexdigest()))
    g=Guard();g.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            fs=[pool.submit(g.job,f'masked_P1_seed{s}_legacy',gpu,'corrected_legacy_eval.py',['--seed',str(s)]) for s,gpu in enumerate([1,2,3])]
            for f in concurrent.futures.as_completed(fs):f.result()
        # Separate final aggregation includes these once-only legacy results.
        runpy.run_path(str(BASE/'mitonet_report_final.py'),run_name='__main__')
        save(BASE/'matched_legacy_queue.json',dict(status='complete',seeds=[0,1,2],resource_peaks=g.peaks))
    except Exception as e:g.stop_all(repr(e));save(BASE/'matched_legacy_queue.json',dict(status='blocked',reason=repr(e)));raise
    finally:g.close()

if __name__=='__main__':main()
