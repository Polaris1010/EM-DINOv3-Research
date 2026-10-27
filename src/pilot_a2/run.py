"""Bounded sequential A2 controller; only this process's children are signalled."""
import fcntl,os,subprocess,time,urllib.request
from common import R,P,read,sha,atomic
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
def website():return urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10).status
def gpu():
    a=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(i):{'uuid':u.strip(),'memory_MB':int(m),'util':int(v)} for i,u,m,v in (x.split(',') for x in a.splitlines())}
def main():
    mapping=gpu();allowed=[i for i in [3,2,1] if mapping[i]['memory_MB']<1024 and mapping[i]['util']==0];assert allowed
    device=allowed[0];assert website()==200
    atomic(R/'evidence/GPU_launch.json',{'physical_GPU':device,'mapping':mapping,'website':200,'GPU0_used':False})
    paths=[P/'samples.json',P/'protocol.json',P/'evaluation_labels.npz']
    for m in ['D2-30k','D2-100k']:
        paths.extend(P/'models'/m/n for n in ['linear_probability.npy','prototype_score.npy','linear_probes.pt','probe_complete.json'])
    before={str(p):sha(p) for p in paths};assert before[str(P/'samples.json')]==read(R/'protocol.json')['samples_sha256']
    atomic(R/'evidence/inputs_before.json',before)
    started=time.monotonic();child=None
    env={**os.environ,'CUDA_VISIBLE_DEVICES':mapping[device]['uuid'],'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(R/'cache/mpl')}
    try:
        for script,args in [('smoke.py',[]),('evaluate.py',['baseline']),('fit.py',[]),('evaluate.py',['final'])]:
            name=script.replace('.py','')+'_'+('_'.join(args) or 'run')
            if script=='fit.py':assert (R/'baseline_complete.json').exists()
            if args==['final']:assert (R/'train_only_bias.json').exists()
            with (R/'logs'/f'{name}.log').open('w') as log:
                child=subprocess.Popen([PY,str(R/'code'/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    assert time.monotonic()-started<2*3600,'A2 bounded2h limit'
                    assert website()==200
                    atomic(R/'STATUS.json',{'status':'running','phase':name,'child_pid':child.pid,'physical_GPU':device,'GPU0_used':False,'seconds':time.monotonic()-started})
                    time.sleep(10)
                assert child.returncode==0,f'{name} failed; no automatic altered protocol'
        after={p:sha(p) for p in before};assert before==after
        atomic(R/'evidence/inputs_after.json',after)
        atomic(R/'STATUS.json',{'status':'complete','seconds':time.monotonic()-started,'physical_GPU':device,'GPU0_used':False,'website_http':website(),'historical_inputs_unchanged':True,'encoder_forward_calls':0,'network_updates':0,'further_experiments_started':False})
    except BaseException as e:
        atomic(R/'STATUS.json',{'status':'failed','error':repr(e),'further_experiments_started':False});raise
    finally:
        if child is not None and child.poll() is None:child.terminate()
if __name__=='__main__':
    with (R/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (R/'STATUS.json').exists(),'Never duplicate existing A2'
        main()
