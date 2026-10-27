"""Bounded Pilot A controller. Own children only; known website context is preserved."""
import fcntl
import os
import subprocess
import sys
import time
import urllib.request
from common import R,PY,read,atomic,sha


def website():
    assert urllib.request.urlopen('http://127.0.0.1:7860/',timeout=10).status==200


def gpus():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    return {int(a):{'uuid':b.strip(),'used_MB':int(c),'utilization':int(d)} for a,b,c,d in (line.split(',') for line in lines)}


class Controller:
    def __init__(self):
        self.started=time.monotonic();self.phase='initialize';self.children=[];self.logs=[]
        self.mapping=gpus();self.allowed=[i for i in [3,2,1] if self.mapping[i]['used_MB']<1024 and self.mapping[i]['utilization']==0]
        assert self.allowed,'No confirmed idle allowed GPU; do not interfere with unknown jobs'
        website();atomic(R/'evidence/launch_GPU_snapshot.json',{'allowed':self.allowed,'GPU_mapping':self.mapping,'website_http':200,'GPU0_used':False})

    def launch(self,script,args=(),gpu=None):
        if gpu is not None:
            live=gpus()[gpu]
            assert live['used_MB']<1024 and live['utilization']==0, 'GPU no longer idle before job launch'
        env={**os.environ,'PILOT_A_ROOT':str(R),'CUDA_VISIBLE_DEVICES':self.mapping[gpu]['uuid'] if gpu is not None else '',
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
             'CUPY_CACHE_DIR':str(R/'cache/cupy'),'CUDA_CACHE_PATH':str(R/'cache/cuda'),'MPLCONFIGDIR':str(R/'cache/matplotlib')}
        name=script.replace('.py','')+'_'+('_'.join(map(str,args)) or str(time.time_ns()))
        handle=(R/'logs'/(name+'.log')).open('a');self.logs.append(handle)
        p=subprocess.Popen([PY,str(R/'code'/script),*map(str,args)],env=env,stdout=handle,stderr=subprocess.STDOUT)
        self.children.append(p);return p

    def health(self):
        assert time.monotonic()-self.started<8*3600,'Bounded Pilot A8h budget; no automatic new phase'
        website()
        atomic(R/'STATUS.json',{'status':'running','phase':self.phase,'seconds':time.monotonic()-self.started,
               'own_children':[p.pid for p in self.children if p.poll() is None],'physical_gpus':self.allowed,'GPU0_used':False,'website_http':200})

    def wait(self,jobs):
        while True:
            self.health()
            for p in jobs:assert p.poll() in (None,0),f'Pilot child{p.pid} failed; inspect logs, no altered experiment'
            if all(p.poll()==0 for p in jobs):return
            time.sleep(10)

    def run(self):
        self.phase='freeze_coordinates_and_verify_assets';self.wait([self.launch('plan.py')])
        self.phase='GPU_numerical_smoke';self.wait([self.launch('smoke.py',gpu=self.allowed[0])])
        self.phase='prepare_images_and_train_GT_only';self.wait([self.launch('data_stage.py',['prep'],self.allowed[0])])
        self.phase='frozen_features_and_train_only_linear_probes'
        pending=[x['id'] for x in read(R/'protocol.json')['models']];active={}
        while pending or active:
            for gpu,p in list(active.items()):
                if p.poll() is not None:
                    assert p.returncode==0,f'Model worker{p.pid} failed'
                    del active[gpu]
            for gpu in self.allowed:
                if gpu not in active and pending:active[gpu]=self.launch('model_stage.py',['--model',pending.pop(0)],gpu)
            self.health();time.sleep(10)
        self.phase='offline_GT_and_existing_Primus_predictions';self.wait([self.launch('data_stage.py',['offline'],self.allowed[0])])
        self.phase='GPU_spectral_separability_and_periodicity_statistics';self.wait([self.launch('analyze.py',gpu=self.allowed[0])])
        self.phase='report_and_figures';self.wait([self.launch('report.py')])
        for spec in read(R/'verified_models.json'):
            assert sha(spec['checkpoint_path'])==spec['sha256'],'Historical asset changed'
        website();atomic(R/'STATUS.json',{'status':'complete','models':5,'layers':[5,11,17,23],'linear_probes':20,
            'SSL_or_Primus_training':False,'Pilot_B_or_C':False,'GPU0_used':False,'website_http':200,
            'historical_checkpoints_unchanged':True,'report':'Feature_Mechanism_Pilot_Report.md','further_experiments_started':False})


if __name__=='__main__':
    R.mkdir(parents=True,exist_ok=True);(R/'logs').mkdir(exist_ok=True)
    with (R/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);controller=None
        try:
            assert not (R/'samples.json').exists(),'Never duplicate an existing Pilot A'
            controller=Controller();controller.run()
        except BaseException as error:
            atomic(R/'STATUS.json',{'status':'failed','phase':controller.phase if controller else 'initialize','error':repr(error),'further_experiments_started':False})
            raise
        finally:
            if controller:
                # Do not enumerate or signal any process that this controller did not launch.
                for child in controller.children:
                    if child.poll() is None:child.terminate()
                for handle in controller.logs:handle.close()
