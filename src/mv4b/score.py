"""Same valid GT, plus output-path foreground retention; no parameter selection."""
import argparse,math,os,sys,time
import numpy as np
from shared import R,M4,MV2,MODEL,read,sha,atomic,rows,evaluation_arrays,config,output,gpu_guard
def counts(a,b,v):return np.array([(a&b&v).sum(),(a&~b&v).sum(),(~a&b&v).sum(),(~a&~b&v).sum()],np.int64)
def metrics(cc):
 tp,fp,fn,tn=map(int,cc);n=tp+fp+fn+tn
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,valid_voxels=n,gt_foreground_voxels=tp+fn,prediction_foreground_voxels=tp+fp,
 dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,predicted_foreground_fraction=(tp+fp)/n)
def score(region,distances=False):
 c=config();row=next(r for r in rows() if r['id']==region);gt,valid,meta=evaluation_arrays(row)
 out=output();infer=read(out/(region+'_inference.json'));assert infer['protocol_sha256']==sha(R/'MV4B_protocol.json')
 sem=np.load(out/(region+'_prediction.npy'),mmap_mode='r');raw=np.load(out/(region+'_raw_threshold.npy'),mmap_mode='r')
 pan=np.load(M4/'runs/Official-MitoNet-v1'/(region+'_prediction.npy'),mmap_mode='r')
 assert sem.shape==gt.shape==raw.shape==pan.shape
 cc={k:np.zeros(4,np.int64) for k in ['semantic','raw_threshold','panoptic']};loss=np.zeros(4,np.int64);plane=[]
 for z in range(gt.shape[0]):
  v=valid[z]>0 if valid is not None else np.ones(gt[z].shape,bool);b=gt[z]>0;s=sem[z]>0;p=pan[z]>0;rawz=raw[z]>0
  for name,a in [('semantic',s),('panoptic',p),('raw_threshold',rawz)]:cc[name]+=counts(a,b,v)
  ls=int((s&~p&v).sum());add=int((p&~s&v).sum());nsem=int((s&v).sum());losttp=int((s&~p&b&v).sum())
  loss+=np.array([ls,add,nsem,losttp]);plane.append({'z_local':z,'semantic_foreground':nsem,'semantic_lost_in_panoptic':ls,'panoptic_outside_semantic':add,'detected_centers':infer['detected_centers_per_plane'][z]})
  if nsem:plane[-1]['lost_fraction']=ls/nsem
 m={k:metrics(v) for k,v in cc.items()};old=read(M4/'runs/Official-MitoNet-v1'/(region+'_metrics.json'))
 assert m['semantic']['valid_voxels']==meta['valid_voxels']
 for k in ['tp','fp','fn','tn','valid_voxels','gt_foreground_voxels']:assert m['panoptic'][k]==old[k]
 lost,extra,nsem,losttp=map(int,loss)
 result={'stage':'MV-4B','model':MODEL,'region_id':region,'source_id':row['source_id'],'volume_id':row['volume_id'],
 'semantic':m['semantic'],'raw_threshold':m['raw_threshold'],'panoptic':m['panoptic'],'semantic_minus_panoptic_dice':m['semantic']['dice']-m['panoptic']['dice'],
 'semantic_foreground_lost_in_panoptic':lost,'lost_semantic_foreground_fraction':lost/nsem if nsem else None,'panoptic_foreground_outside_semantic':extra,
 'semantic_true_positive_lost_in_panoptic':losttp,'lost_planes':sum(q['semantic_lost_in_panoptic']>0 for q in plane),
 'zero_center_planes':sum(q['detected_centers']==0 for q in plane),'plane_stats':plane,
 'valid_mask_policy':row['valid_mask_policy'],'spacing_status':row['spacing_status'],'spacing_zyx_nm':row['spacing_zyx_nm'],
 'protocol_sha256':sha(R/'MV4B_protocol.json'),'manifest_sha256':c['manifest_sha256'],'checkpoint_sha256':infer['checkpoint_sha256'],'tta':False,
 'physical_gpu':infer['physical_gpu'],'old_panoptic_metrics_unchanged':True,'distance_metrics_computed':False}
 if distances:
  gpu=gpu_guard();os.environ['MV2_GPU_EDT']='1';sys.path.insert(0,str(MV2/'src'));import validation as evaluator;import gpu_edt
  tp,fp,fn=m['semantic']['tp'],m['semantic']['fp'],m['semantic']['fn'];t0=time.time()
  if tp+fn==0:hd=0. if tp+fp==0 else math.inf;ns=1. if tp+fp==0 else 0.
  elif tp+fp==0:hd=math.inf;ns=0.
  else:hd,ns=evaluator.distances(sem>0,gt>0,valid>0 if valid is not None else np.ones(gt.shape,bool),row['spacing_zyx_nm'])
  result['semantic'].update(hd95_nm=hd,nsd_16nm=ns);result.update(distance_metrics_computed=True,distance_physical_gpu=gpu,distance_evaluator_source_sha256=sha(MV2/'src/validation.py'),metric_seconds=time.time()-t0);gpu_edt.release()
 atomic(out/(region+('_metrics.json' if distances else '_path_comparison.json')),result)
 print('SEMANTIC_SCORED',region,m['semantic']['dice'],result['lost_semantic_foreground_fraction'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--region',required=True);p.add_argument('--distances',action='store_true');a=p.parse_args();score(a.region,a.distances)
