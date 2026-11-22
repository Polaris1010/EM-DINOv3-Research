"""Bind public CEM-MitoLab assets to the author's supervised training entry point."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import socket
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'empanada-main'
DATA = Path('/opt/data/3_117/tanghr/MitoNet/dataset/EMPIAR-11037/cem_mitolab')
WEIGHT = ROOT/'swav.pth.tar'

def local_tracking():
    # Replace the external tracking service only. Training logic is upstream.
    log = types.ModuleType('mlflow')
    folder = ROOT/'mlruns/paper'
    folder.mkdir(parents=True,exist_ok=True)
    def record(k,v,**kw):
        with (folder/'events.jsonl').open('a') as f:
            f.write(json.dumps([k,v,kw],default=str)+'\n')
    log.log_param = log.log_metric = record
    log.end_run = lambda: None
    log.set_experiment = lambda name: record('experiment',name)
    log.active_run = lambda: types.SimpleNamespace(info=types.SimpleNamespace(run_id='paper'))
    log.get_artifact_uri = lambda: folder.as_uri()
    sys.modules['mlflow']=log

def run():
    import torch,yaml
    assert socket.gethostname().split('.')[0]=='mira-4090-8'
    ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuids={int(i):u.strip() for i,u in (s.split(',') for s in ids.splitlines())}
    assert os.environ['CUDA_VISIBLE_DEVICES'].split(',')==[uuids[i] for i in (1,2,3)]
    assert torch.cuda.device_count()==3
    assert hashlib.md5(WEIGHT.read_bytes()).hexdigest()=='8ed17e0979420e01839f459a24aca5bf'
    sys.path.insert(0,str(SRC))
    import empanada.data._base as base
    orig_glob=base.glob
    # Pair image and instance TIFFs by name independently of directory listing order.
    base.glob=lambda pattern: sorted(p for p in orig_glob(pattern) if Path(p).suffix.lower() in ('.tif','.tiff'))
    counts={}
    for collection in sorted(DATA.iterdir()):
        if not collection.is_dir(): continue
        images={p.name for p in (collection/'images').glob('*.tif*')}
        masks={p.name for p in (collection/'masks').glob('*.tif*')}
        assert images and images==masks, collection
        counts[collection.name]=len(images)
    (ROOT/'collection_counts.json').write_text(json.dumps(counts,indent=2))
    cfg=yaml.safe_load((SRC/'projects/mitonet/configs/mmm_panoptic_deeplab_pointrend.yaml').read_text())
    t=cfg['TRAIN']
    t.update(train_dir=str(DATA),additional_train_dirs=None,model_dir=str(ROOT/'models'),
             encoder_pretraining=str(WEIGHT),batch_size=64,print_freq=25)
    t['schedule_params']['epochs']=120
    t['dataset_params']['weight_gamma']=0.3
    cfg['EVAL']['eval_dir']=None
    # Verify that the author's prefix mapping actually initializes the entire encoder.
    from empanada import models
    state=torch.load(WEIGHT,map_location='cpu',weights_only=False)
    enc=models.__dict__[cfg['MODEL']['arch']](**cfg['MODEL']).encoder
    expected=enc.state_dict()
    assert all(k in state['state_dict'] and state['state_dict'][k].shape==v.shape for k,v in expected.items()), 'incomplete SwAV encoder'
    del enc,expected,state
    config=ROOT/'mitonet_paper.yaml'
    config.write_text(yaml.safe_dump(cfg))
    print('collections',len(counts),'images',sum(counts.values()),'batch',t['batch_size'],flush=True)
    os.chdir(ROOT)
    local_tracking()
    sys.argv=[str(SRC/'scripts/train.py'),str(config)]
    runpy.run_path(sys.argv[0],run_name='__main__')

if __name__=='__main__':
    run()
