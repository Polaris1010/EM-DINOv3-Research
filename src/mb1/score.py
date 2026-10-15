"""Score original GT only after fixed inference; unchanged MONAI distances in2D."""
import argparse,math,sys,time
import numpy as np
from common import R,OLD,MV2,read,sha,atomic,rows,run_spec,Planes
from metrics import METRICS,overlap,average,distribution
def main(run_id):
 sys.path[:0]=[str(MV2/'src'),str(OLD/'src')]
 import validation as evaluator
 spec=run_spec(run_id);out=R/'runs'/run_id;images=[];datasets=[];started=time.time()
 for row in rows():
  infer=read(out/(row['id']+'_inference.json'));assert infer['all_native_planes'] and not infer['gt_read_by_inference'] and not infer['tta']
  assert infer['protocol_sha256']==sha(R/'MB1_protocol.json') and infer['manifest_sha256']==sha(R/'MB1_benchmark_manifest.json')
  assert infer['checkpoint_sha256']==spec['checkpoint_sha256']
  gt=Planes(row['label_path'],row['shape_zyx']);pred=np.load(out/(row['id']+'_prediction.npy'),mmap_mode='r');assert list(pred.shape)==row['shape_zyx']
  for z in range(row['planes']):
   raw=np.asarray(gt[z]);assert raw.min()>=0,'Official GT negative/ignore semantics must not be silently discarded'
   a=pred[z]>0;b=raw>0;assert a.shape==b.shape
   tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum());tn=int((~a&~b).sum());m=overlap(tp,fp,fn,tn)
   spacing=row['spacing_yx_nm']
   if spacing is None:hd=ns=None
   elif tp+fn==0:hd=0. if tp+fp==0 else math.inf;ns=1. if tp+fp==0 else 0.
   elif tp+fp==0:hd=math.inf;ns=0.
   else:hd,ns=evaluator.distances(a,b,np.ones(a.shape,bool),spacing)
   images.append({'run_id':run_id,'dataset_id':row['dataset_id'],'z':z,'image_id':row.get('image_id',''),'dice':m['dice'],'iou':m['iou'],'precision':m['precision'],'recall':m['recall'],'hd95_nm':hd,'nsd_16nm':ns,'predicted_foreground_fraction':m['predicted_foreground_fraction'],'tp':tp,'fp':fp,'fn':fn,'tn':tn,'entry_id':row['id']})
   if z%50==0:atomic(out/'score_progress.json',{'run_id':run_id,'dataset':row['dataset'],'entry_id':row['id'],'z':z+1,'planes':row['planes'],'scored_images':len(images),'seconds':time.time()-started});print('MB1_SCORE',run_id,row['id'],z+1,flush=True)
  gt.close();del pred
 for bid in range(1,8):
  selected=[r for r in images if r['dataset_id']==bid];d=next(r for r in rows() if r['dataset_id']==bid);counts={k:sum(r[k] for r in selected) for k in ['tp','fp','fn','tn']};pooled=overlap(**counts)
  result={'run_id':run_id,'model':spec['model'],'seed':spec['seed'],'dataset_id':bid,'dataset':d['dataset'],'planes':len(selected),'entries':len({r['entry_id'] for r in selected}),**{k:average([r[k] for r in selected]) for k in METRICS},**counts,'pooled_dice':pooled['dice'],'pooled_iou':pooled['iou'],'volumetric_semantic_iou':pooled['iou'] if d['type']=='volume' else None,'calibrated_planes':sum(r['hd95_nm'] is not None for r in selected),'empty_gt_planes':sum(r['tp']+r['fn']==0 for r in selected),'empty_prediction_planes':sum(r['tp']+r['fp']==0 for r in selected),'precision_undefined_planes':sum(r['precision'] is None for r in selected),'recall_undefined_planes':sum(r['recall'] is None for r in selected),'checkpoint_sha256':spec['checkpoint_sha256'],'protocol_sha256':sha(R/'MB1_protocol.json'),'manifest_sha256':sha(R/'MB1_benchmark_manifest.json'),'distance_evaluator_source_sha256':sha(MV2/'src/validation.py')};datasets.append(result)
 assert len(images)==2595 and len(datasets)==7
 atomic(out/'metrics.json',{'status':'complete','run_id':run_id,'images':images,'datasets':datasets,'tem_distributions':{k:distribution([r[k] for r in images if r['dataset_id']==7]) for k in METRICS},'seconds':time.time()-started,'checkpoint_sha256':spec['checkpoint_sha256'],'protocol_sha256':sha(R/'MB1_protocol.json')})
 print('MB1_ALL_SCORED',run_id,len(images),time.time()-started,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);main(p.parse_args().run_id)
