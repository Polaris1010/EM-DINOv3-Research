"""Score the MitoNet instance foreground with the unchanged MV2 physical evaluator."""
import argparse,math,os,sys,time
from pathlib import Path
import numpy as np
from common import R,MV2,MODEL,read,sha,atomic,rows,evaluation_arrays,config,gpu_guard

def score(region):
 gpu=gpu_guard();os.environ['MV2_GPU_EDT']='1';sys.path.insert(0,str(MV2/'src'))
 import validation as evaluator
 import gpu_edt
 out=R/'runs'/MODEL;path=out/(region+'_metrics.json')
 if path.exists():assert read(path)['protocol_sha256']==sha(R/'MV4_protocol.json');return
 row=next(r for r in rows() if r['id']==region);gt,valid,j=evaluation_arrays(row)
 infer=read(out/(region+'_inference.json'));assert infer['status']=='complete' and infer['all_native_z_inferred']
 assert infer['protocol_sha256']==sha(R/'MV4_protocol.json')
 pred=np.load(out/(region+'_prediction.npy'),mmap_mode='r');assert pred.shape==gt.shape
 counts=np.zeros(4,np.int64)
 for z in range(gt.shape[0]):
  a=pred[z]>0;b=gt[z]>0;v=valid[z]>0 if valid is not None else np.ones(a.shape,bool)
  counts+=np.array([(a&b&v).sum(),(a&~b&v).sum(),(~a&b&v).sum(),(~a&~b&v).sum()],np.int64)
 tp,fp,fn,tn=map(int,counts);n=tp+fp+fn+tn;assert n==j['valid_voxels']
 t0=time.time()
 if tp+fn==0:hd=0. if tp+fp==0 else math.inf;ns=1. if tp+fp==0 else 0.
 elif tp+fp==0:hd=math.inf;ns=0.
 else:hd,ns=evaluator.distances(pred>0,gt>0,valid>0 if valid is not None else np.ones(gt.shape,bool),row['spacing_zyx_nm'])
 result={'stage':'MV-4','model':MODEL,'region_id':region,'source_id':row['source_id'],'volume_id':row['volume_id'],
  'dev_kind':row['dev_kind'],'acquisition_id':row['acquisition_id'],'spacing_status':row['spacing_status'],
  'spacing_zyx_nm':row['spacing_zyx_nm'],'valid_mask_policy':row['valid_mask_policy'],
  'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
  'hd95_nm':hd,'nsd_16nm':ns,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'valid_voxels':n,
  'gt_foreground_voxels':tp+fn,'prediction_foreground_voxels':tp+fp,'predicted_foreground_fraction':(tp+fp)/n,
  'protocol_sha256':sha(R/'MV4_protocol.json'),'manifest_sha256':config()['manifest_sha256'],
  'checkpoint_sha256':infer['checkpoint_sha256'],'tta':False,'physical_gpu':gpu,
  'distance_evaluator_source_sha256':sha(MV2/'src/validation.py'),'distance_backend':'MV2_existing_exact_EDT_MONAI','metric_seconds':time.time()-t0}
 atomic(path,result);gpu_edt.release();print('MITONET_SCORED',region,result['dice'],hd,ns,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--region',required=True);a=p.parse_args();score(a.region)
