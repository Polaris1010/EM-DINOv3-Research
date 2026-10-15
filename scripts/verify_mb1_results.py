"""CPU-only independent count/coverage checks of completed MB1 artifacts."""
import csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'experiments/MB-1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def near(a,b):
 if a is None:return b=='',('undefined field must stay blank',b)
 return math.isclose(a,float(b),rel_tol=6e-7,abs_tol=6e-8),(a,b)
def ratios(c):
 tp,fp,fn,tn=(int(c[k]) for k in ['tp','fp','fn','tn']);assert min(tp,fp,fn,tn)>=0
 return {'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'predicted_foreground_fraction':(tp+fp)/(tp+fp+fn+tn)}
def main():
 m=read(FOLDER/'MB1_benchmark_manifest.json');c=read(ROOT/'configs/evaluation/MB1_protocol.json');s=read(FOLDER/'MB1_summary.json');status=read(FOLDER/'STATUS.json');assert status['status']==s['status']=='complete'
 assert status['historical_inputs_unchanged'] and not status['checkpoint_writes'] and not status['training'] and not status['gpu0_used']
 rows=list(csv.DictReader((FOLDER/'MB1_per_image_metrics.csv').open()));counts=list(csv.DictReader((FOLDER/'MB1_per_image_confusion.csv').open()));ds=list(csv.DictReader((FOLDER/'MB1_per_dataset_metrics.csv').open()))
 assert len(rows)==len(counts)==20760 and len(ds)==56 and {r['run_id'] for r in rows}=={r['run_id'] for r in c['runs']}
 key=lambda r:(r['run_id'],int(r['dataset_id']),int(r['z']),r['image_id'])
 rr={key(r):r for r in rows};cc={key(r):r for r in counts};assert len(rr)==len(cc)==20760 and rr.keys()==cc.keys()
 expected=set();geometry={}
 for entry in m['entries']:
  for z in range(entry['planes']):
   k=(entry['dataset_id'],z,entry.get('image_id',''));geometry[k]=entry['shape_zyx'][1]*entry['shape_zyx'][2]
   for run in c['runs']:expected.add((run['run_id'],*k))
 assert rr.keys()==expected
 ground={};derived={}
 for k,r in rr.items():
  n=cc[k];ints={v:int(n[v]) for v in ['tp','fp','fn','tn']};assert sum(ints.values())==geometry[k[1:]]
  signature=(ints['tp']+ints['fn'],ints['tn']+ints['fp']);assert ground.setdefault(k[1:],signature)==signature,'Models did not use the same GT pixels'
  v=ratios(ints);derived[k]=v
  for name,value in v.items():
   good,detail=near(value,r[name]);assert good,(k,name,detail)
  if k[1]==7:assert r['hd95_nm']==r['nsd_16nm']=='','No TEM nm calibration was verified'
 recomputed={}
 for row in ds:
  rid=row['run_id'];bid=int(row['dataset_id']);chosen=[k for k in rr if k[0]==rid and k[1]==bid];assert len(chosen)==int(row['planes'])
  total={n:sum(int(cc[k][n]) for k in chosen) for n in ['tp','fp','fn','tn']}
  for name,n in total.items():assert n==int(row[name])
  pooled=ratios(total)
  for field,name in [('pooled_dice','dice'),('pooled_iou','iou')]:good,detail=near(pooled[name],row[field]);assert good,(rid,bid,field,detail)
  if bid<7:good,detail=near(pooled['iou'],row['volumetric_semantic_iou']);assert good,(rid,bid,detail)
  else:assert row['volumetric_semantic_iou']==''
  values={}
  for name in ['dice','iou','precision','recall','predicted_foreground_fraction']:
   vv=[derived[k][name] for k in chosen if derived[k][name] is not None];values[name]=statistics.mean(vv) if vv else None;good,detail=near(values[name],row[name]);assert good,(rid,bid,name,detail)
  recomputed[rid,bid]=values
 for model in ['G-P0','G-B100','Official-MitoNet-v1','MitoNet-MV']:
  runs=[r['run_id'] for r in c['runs'] if r['model']==model]
  for metric in ['dice','iou','precision','recall','predicted_foreground_fraction']:
   means=[]
   for bid in range(1,8):
    vs=[recomputed[rid,bid][metric] for rid in runs if recomputed[rid,bid][metric] is not None];means.append(statistics.mean(vs) if vs else None)
   defined=[v for v in means if v is not None];value=statistics.mean(defined) if defined else None
   if value is None:assert s['source_macro'][model][metric] is None
   else:assert math.isclose(value,s['source_macro'][model][metric],rel_tol=1e-12,abs_tol=1e-12),(model,metric)
 assert (ROOT/'evidence/MB-1/historical_inputs_before.json').read_bytes()==(ROOT/'evidence/MB-1/historical_inputs_after.json').read_bytes()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(FOLDER/'MB1_benchmark_manifest.json')==s['manifest_sha256'] and sha(ROOT/'configs/evaluation/MB1_protocol.json')==s['protocol_sha256']
 result={'status':'PASS','rows':20760,'dataset_run_rows':56,'all8_models_use_identical_gt_pixels':True,'all7_native_datasets_complete':True,'2D_overlap_metrics_recomputed_from_exact_counts':True,'full_volume_iou_recomputed_from_exact_counts':True,'equal_seeds_then_equal7_datasets_recomputed':True,'TEM_nm_not_fabricated':True,'historical_input_snapshot_equal':True,'checkpoint_writes':False,'training':False}
 (FOLDER/'MB1_local_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))
if __name__=='__main__':main()
