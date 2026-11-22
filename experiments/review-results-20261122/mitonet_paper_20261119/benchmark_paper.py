"""Bind EMPIAR-10982 TIFFs to the author volume evaluator and score TEM by image."""
from pathlib import Path
import csv
import functools
import json
import os
import runpy
import statistics
import subprocess
import sys

import numpy as np
import torch
import yaml
import zarr
import tifffile
from skimage.measure import regionprops
from train_paper import ROOT, SRC, local_tracking
sys.path.insert(0,str(SRC))
from empanada.array_utils import rle_encode
from empanada.inference.tracker import InstanceTracker
from empanada.evaluation import Evaluator,f1_50,f1_75,panoptic_quality
from empanada.evaluation.instance_metrics import ap

BENCH=ROOT/'benchmark'
MANIFEST=ROOT.parent/'mb1_official_benchmark_20261015/MB1_benchmark_manifest.json'
PAPER={'c_elegans':(3,500,1),'fly_brain':(3,500,2),'hela_cell':(5,800,2),
       'glycolytic_muscle':(5,3000,2),'lucchi_pp':(7,5000,2),'salivary_gland':(7,1000,1)}
FIELDS=['IoU','F1@50','F1@75','AP@50','AP@75','PQ']

def add_plane(tracker, mask, index):
    props={}
    for rp in regionprops(mask):
        positions=np.ravel_multi_index(rp.coords.T,mask.shape)
        starts,runs=rle_encode(positions)
        props[int(rp.label)]=dict(box=tuple(map(int,rp.bbox)),starts=starts,runs=runs)
    tracker.update(props,index)

def prepare(row):
    name=row['dataset']
    path=BENCH/(name+'.zarr')
    if (path/'input_complete.json').exists():return path
    shape=tuple(row['shape_zyx'])
    group=zarr.open_group(str(path),mode='a',zarr_format=2)
    image=group.create_array('em',shape=shape,dtype=np.uint8,chunks=(16,128,128),overwrite=True)
    tracker=InstanceTracker(1,20000,shape,'xy')
    with tifffile.TiffFile(row['image_path']) as a,tifffile.TiffFile(row['label_path']) as b:
        # Some public TIFFs store contiguous volumes in a single series.
        def stack(tf):
            try:return tifffile.memmap(tf.filehandle.path)
            except ValueError:return tf.asarray(out='memmap')
        raw,ids=stack(a),stack(b)
        assert raw.shape==ids.shape==shape and raw.dtype==np.uint8
        for start in range(0,shape[0],16):
            slab=raw[start:start+16]
            image[start:start+len(slab)]=slab
            assert np.array_equal(image[start:start+len(slab)],slab)
            for z in range(start,start+len(slab)):add_plane(tracker,ids[z],z)
    tracker.write_to_json(path/'mito_gt.json')
    (path/'input_complete.json').write_text(json.dumps(row))
    return path

def score(gt,pred):
    evaluator=Evaluator(None,{'F1@50':f1_50,'F1@75':f1_75,'AP@50':functools.partial(ap,iou_thr=.5),
                              'AP@75':functools.partial(ap,iou_thr=.75)}, {'PQ':panoptic_quality})
    return evaluator(gt,pred)

def overlap(mask,pred):
    a,b=mask>0,pred>0
    return int(np.count_nonzero(a&b)),int(np.count_nonzero(a|b))

def volume(row):
    from empanada.config_loaders import load_config
    name=row['dataset']
    p=prepare(row)
    kernel,size,votes=PAPER[name]
    cfg=load_config(str(SRC/'projects/mitonet/configs/mmm_median_inference.yaml'))
    cfg['engine_params'].update(median_kernel_size=kernel,confidence_thr=.3,nms_kernel=7)
    cfg['consensus_params']['pixel_vote_thr']=votes
    cfg['filters']=[dict(name='remove_small_objects',min_size=size),dict(name='remove_pancakes',min_span=8)]
    settings=BENCH/(name+'_inference.yaml')
    settings.write_text(yaml.safe_dump(cfg))
    src=SRC/'projects/mitonet/scripts/evaluate3d.py'
    # zarr 3 requires explicit chunks. The upstream prediction code is unchanged.
    code=src.read_text().replace('chunks=(1, None, None)','chunks=(1, shape[1], shape[2])')
    sys.argv=[str(src),str(ROOT/'mitonet_paper.yaml'),str(settings),str(p)]
    local_tracking()
    exec(compile(code,str(src),'exec'),{'__name__':'__main__','__file__':str(src)})
    metrics=score(p/'mito_gt.json',p/'mitonet_paper_mito_pred.json')
    pred=zarr.open_group(str(p),mode='r')['mitonet_paper_mito_pred']
    gt=tifffile.imread(row['label_path'],out='memmap')
    pairs=[overlap(gt[i],pred[i]) for i in range(len(gt))]
    inter,union=map(sum,zip(*pairs))
    metrics['IoU']=inter/union if union else 1.
    return metrics

def tem(rows):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from empanada import models
    from empanada.inference.engines import PanopticDeepLabEngine
    from empanada.inference.postprocess import factor_pad
    cfg=yaml.safe_load((ROOT/'mitonet_paper.yaml').read_text())
    state=torch.load(ROOT/'models/mitonet_paper_checkpoint.pth.tar',map_location='cpu',weights_only=False)
    model=models.PanopticDeepLabPR(**cfg['MODEL'])
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state['state_dict'].items()},strict=True)
    engine=PanopticDeepLabEngine(model.cuda(),**cfg['EVAL']['engine_params'])
    transform=A.Compose([A.Normalize(**state['norms']),ToTensorV2()])
    out=BENCH/'TEM-100'
    out.mkdir(exist_ok=True)
    scores=[]
    for row in rows:
        img=tifffile.imread(row['image_path']).squeeze()
        gt=tifffile.imread(row['label_path']).squeeze()
        h,w=gt.shape
        x=transform(image=img[...,None])['image'][None]
        pred=engine(factor_pad(x,128)).squeeze().cpu().numpy()[:h,:w]
        assert img.shape==gt.shape==pred.shape
        paths=[]
        for kind,m in [('gt',gt),('pred',pred)]:
            tracker=InstanceTracker(1,1000,(1,h,w),'xy')
            add_plane(tracker,m,0)
            path=out/(row['id']+'_'+kind+'.json')
            tracker.write_to_json(path)
            paths.append(path)
        inter,union=overlap(gt,pred)
        scores.append(dict(id=row['id'],IoU=inter/union if union else 1.,**score(*paths)))
    assert len(scores)==100
    (out/'per_image.json').write_text(json.dumps(scores,indent=2))
    return {m:statistics.mean(r[m] for r in scores) for m in FIELDS}

if __name__=='__main__':
    BENCH.mkdir(exist_ok=True)
    rows=json.loads(MANIFEST.read_text())['entries']
    mode=sys.argv[1]
    if mode=='prepare':
        for r in rows:
            if r['dataset'] in PAPER:prepare(r)
    else:
        # Each launched process receives one allowed physical GPU by UUID.
        allowed=subprocess.check_output(['nvidia-smi','-i','1,2,3','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
        assert os.environ.get('CUDA_VISIBLE_DEVICES') in allowed
        selected=[r for r in rows if r['dataset']==mode]
        metrics=tem(selected) if mode=='TEM-100' else volume(selected[0])
        (BENCH/(mode+'_metrics.json')).write_text(json.dumps(metrics,indent=2))
