"""Separate image/train-label preparation from final offline evaluation-label access."""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import R, MV2, Data, read, atomic, sha, npy, prime_prediction


def pool(x, size=16):
    return F.avg_pool2d(torch.as_tensor(np.ascontiguousarray(x), device='cuda', dtype=torch.float32)[None,None],size,size)[0,0]


def prep():
    items=read(R/'samples.json');data=Data();n=len(items)
    images=np.lib.format.open_memmap(R/'images.npy',mode='w+',dtype=np.float16,shape=(n,256,256))
    labels=np.zeros((n,256),np.float32);valid=np.zeros((n,256),bool)
    for item in items:
        images[item['id']]=data.tile(item,'image')
        if item['role']=='train':
            labels[item['id']]=pool(data.tile(item,'label')>0).flatten().cpu().numpy()
            valid[item['id']]=(pool(data.tile(item,'valid'))>=.95).flatten().cpu().numpy()
    images.flush();np.savez(R/'train_labels.npz',fraction=labels,valid=valid)
    atomic(R/'evidence/train_data_isolation.json',{'status':'PASS','training_region_ids':sorted({i['region_id'] for i in items if i['role']=='train'}),
        'evaluation_GT_read':False,'image_count':n,'train_crops':sum(i['role']=='train' for i in items),
        'samples_sha256':sha(R/'samples.json')})


def offline():
    protocol=read(R/'protocol.json');items=read(R/'samples.json')
    for model in protocol['models']:
        result=read(R/'models'/model['id']/'probe_complete.json')
        assert result['status']=='complete' and result['evaluation_GT_read']==False
        assert sha(R/'models'/model['id']/'linear_probes.pt')==result['probe_sha256']
    os.environ.setdefault('CUDA_PATH',str(MV2/'gpu_deps/nvidia/cuda_runtime'))
    sys.path[:0]=[str(MV2/'src'),str(MV2/'gpu_deps')]
    import gpu_edt
    data=Data();eval_items=[i for i in items if i['role']=='eval'];n=len(items)
    y=np.zeros((n,256),np.float32);valid=np.zeros((n,256),bool);far=np.zeros((n,256),bool)
    gt_pixels=np.zeros((n,256,256),bool);valid_pixels=np.zeros_like(gt_pixels);far_pixels=np.zeros_like(gt_pixels)
    predictions={m['id']:np.zeros_like(gt_pixels) for m in protocol['models']}
    input_assets={};last_key=None
    for item in eval_items:
        key=item['region_id'],item['z']
        if key!=last_key:
            full_gt=np.asarray(data.plane(item,'label'))>0
            full_valid=np.asarray(data.plane(item,'valid')).astype(bool)
            distance=gpu_edt.edt_array(~full_gt,item['spacing_yx_nm']) if full_gt.any() else np.full(full_gt.shape,np.inf)
            full_far=(distance>protocol['far_background_xy_nm'])&(~full_gt)&full_valid
            last_key=key
        k=item['id'];sl=(slice(item['y'],item['y']+256),slice(item['x'],item['x']+256))
        def crop(a):
            t=a[sl];return np.pad(t,((0,256-t.shape[0]),(0,256-t.shape[1])),constant_values=False)
        gt_pixels[k]=crop(full_gt);valid_pixels[k]=crop(full_valid);far_pixels[k]=crop(full_far)
        y[k]=pool(gt_pixels[k]).flatten().cpu().numpy()
        valid[k]=(pool(valid_pixels[k])>=.95).flatten().cpu().numpy()
        far[k]=(pool(far_pixels[k])>=.95).flatten().cpu().numpy()&(y[k]==0)
        for model in protocol['models']:
            path=prime_prediction(item,model['id']);assert path.is_file(),str(path)
            st=path.stat();input_assets[str(path)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
            a=npy(path);assert list(a.shape)==item['shape_zyx'],(path,a.shape,item['shape_zyx'])
            predictions[model['id']][k]=crop(a[item['z']]>0)
        atomic(R/'offline_progress.json',{'sample':k,'evaluation_samples':len(eval_items)})
    for model,array in predictions.items():np.save(R/('primus_'+model+'.npy'),array)
    np.savez(R/'evaluation_labels.npz',fraction=y,valid=valid,far=far,GT=gt_pixels,valid_pixels=valid_pixels,far_pixels=far_pixels)
    atomic(R/'evidence/offline_label_access.json',{'status':'PASS','all_probe_hashes_pinned_before_label_read':True,
        'evaluation_GT_used_for_fitting':False,'distance_backend':'verified_exact_GPU_EDT_full_image_XY',
        'far_background_nm':180,'not_3D_distance':True,'prediction_assets':input_assets,'further_training':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prep','offline']);args=parser.parse_args()
    torch.set_num_threads(2)
    (prep if args.mode=='prep' else offline)()
