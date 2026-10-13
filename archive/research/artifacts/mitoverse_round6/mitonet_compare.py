"""Fixed native-grid MitoNet M0: full-volume 2D foreground comparison."""
import os, sys, json, time, argparse, hashlib, fcntl
from pathlib import Path
from settings import BASE, ROOT, R4
sys.path.insert(0,str(R4))
import common as c
import torch, numpy as np, zarr, yaml
OFFICIAL=ROOT/'codex/mitotopo_20260908_round2/official'
sys.path.insert(0,str(OFFICIAL))
from postprocess import factor_pad,get_panoptic_segmentation

@torch.inference_mode()
def infer(net,x):
    h,w=x.shape[-2:]
    if max(h,w)<=512:return net(factor_pad(x,16))
    # Fixed native-grid tiles; offsets are relative vectors and remain translation invariant.
    accum={k:torch.zeros((1,ch,h,w),device='cuda') for k,ch in [('sem_logits',1),('ctr_hmp',1),('offsets',2)]}
    weights=torch.zeros((1,1,h,w),device='cuda')
    def starts(n):
        return [0] if n<=512 else list(range(0,n-512,256))+[n-512]
    yy=torch.arange(512,device='cuda').float()-255.5
    g=torch.exp(-.5*(yy/64)**2);g=(g[:,None]*g[None,:]).clamp_min(1e-7)[None,None]
    for y in starts(h):
        for z in starts(w):
            patch=x[...,y:y+512,z:z+512];ph,pw=patch.shape[-2:];q=net(factor_pad(patch,16));weight=g[...,:ph,:pw]
            for k in accum:accum[k][...,y:y+ph,z:z+pw]+=q[k][...,:ph,:pw]*weight
            weights[...,y:y+ph,z:z+pw]+=weight
    assert (weights>0).all()
    return {k:v/weights for k,v in accum.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=Path(os.environ['RUN_OUTPUT']);torch.set_num_threads(2)
    protocol=json.loads((BASE/'mitonet_comparison_protocol.json').read_text())
    assert c.sha(OFFICIAL/'MitoNet_v1.pth')==protocol['weight_sha256']
    config=yaml.safe_load((OFFICIAL/'MitoNet_v1.yaml').read_text());cfg=config['FINETUNE']['engine_params']
    assert config['norms']=={'mean':.57571,'std':.12765} and config['padding_factor']==16
    net=torch.jit.load(str(OFFICIAL/'MitoNet_v1.pth'),map_location='cpu').eval().cuda()
    rows=json.loads((R4/'split_mitoverse_probe_v1.json').read_text())['volumes']
    ids=protocol['shards'][str(a.shard)]
    if a.smoke:ids=[max(ids,key=lambda v: np.prod(zarr.open_array(rows[v]['image_path'],mode='r').shape[1:]))]
    result={};start=time.time();lock=(BASE/'mitonet_io.lock').open('a')
    for vid in ids:
        row=rows[vid];root=zarr.open_group(str(Path(row['image_path']).parent),mode='r');raw=root['img'];lab=root['mito'];region=root['mask'] if 'mask' in root else None
        assert raw.shape==lab.shape and raw.dtype==np.uint8
        shape=raw.shape;zs=[shape[0]//2] if a.smoke else range(shape[0]);counts=np.zeros(4,np.int64);postcounts=counts.copy();valid_total=0;t0=time.time()
        pred=None
        if not a.smoke:
            pred=zarr.open_group(str(out/'predictions'/f'{vid}.zarr'),mode='w')
            for key,dt in [('probability','float16'),('binary','uint8'),('postprocessed_binary','uint8')]:pred.create_dataset(key,shape=shape,chunks=(1,256,256),dtype=dt)
            pred.attrs.update(dict(grid='original ZYX',threshold=.5,tta=False,axis='XY',input_scale=1,source=row['source'],voxel_nm_zyx=row['voxel_nm_z_y_x'],model='MitoNet_v1',instance_identity='2D only; not linked across slices'))
        for z in zs:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                im=np.asarray(raw[z]);target=np.asarray(lab[z])>0;valid=np.asarray(region[z])!=0 if region is not None else np.ones(im.shape,dtype=bool)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
            x=torch.from_numpy((im.astype(np.float32)/255.-config['norms']['mean'])/config['norms']['std'])[None,None].cuda()
            with torch.inference_mode():
                q=infer(net,x);assert all(torch.isfinite(v).all() for v in q.values())
                prob=q['sem_logits'].sigmoid();assert prob.shape[1]==1
                sem=(prob>=cfg['confidence_thr']).long()
                pan,_=get_panoptic_segmentation(sem,q['ctr_hmp'],q['offsets'],cfg['thing_list'],cfg['label_divisor'],cfg['stuff_area'],cfg['void_label'],cfg['nms_threshold'],cfg['nms_kernel'])
                pr=prob[0,0,:shape[1],:shape[2]].cpu().numpy();binary=pr>=.5
                pb=pan.reshape(-1,*pan.shape[-2:])[0,:shape[1],:shape[2]].cpu().numpy()>0
                if a.smoke:
                    repeat=infer(net,x)['sem_logits'].sigmoid()[0,0,:shape[1],:shape[2]].cpu().numpy()
                    assert np.max(np.abs(repeat-pr))<=1e-5
            assert pr.shape==target.shape==pb.shape
            counts+=c.confusion(binary[valid],target[valid]);postcounts+=c.confusion(pb[valid],target[valid]);valid_total+=int(valid.sum())
            if pred is not None:
                fcntl.flock(lock,fcntl.LOCK_EX)
                try:pred['probability'][z]=pr.astype(np.float16);pred['binary'][z]=binary.astype(np.uint8);pred['postprocessed_binary'][z]=pb.astype(np.uint8)
                finally:fcntl.flock(lock,fcntl.LOCK_UN)
            if z%25==0:c.save(out/'progress.json',dict(volume=vid,z=z,depth=shape[0],completed_volumes=len(result),seconds=time.time()-start));print(vid,z,flush=True)
        assert sum(counts)==valid_total and sum(postcounts)==valid_total
        result[vid]=dict(source=row['source'],split='legacy_dev_test' if row['split'].startswith('test') else row['split'],**c.metrics(counts),postprocessed_binary=c.metrics(postcounts),valid_voxels=valid_total,slices=len(zs),seconds=time.time()-t0)
        c.save(out/'volume_metrics_partial.json',result)
    c.save(out/'result.json',dict(status='complete',model='MitoNet_v1',protocol='fixed native-grid 2D M0, 512 tiles stride256 if needed; no target fine-tuning, no scale/threshold search',smoke=a.smoke,volumes=result,seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),weight_sha256=protocol['weight_sha256']))
    print('MITONET_COMPLETE',a.shard,'smoke',a.smoke,flush=True)

if __name__=='__main__':main()
