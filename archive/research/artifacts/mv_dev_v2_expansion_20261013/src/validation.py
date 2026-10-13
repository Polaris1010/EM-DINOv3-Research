"""Existing frozen Primus inference on the full usable development union. No training."""
from pathlib import Path
import sys,json,hashlib,collections,csv,math,time,os,gc,argparse,statistics
R=Path(__file__).resolve().parents[1]
OLD=R.parent/'mv_generalist_stage1_20261013'
sys.path[:0]=[str(OLD/'src'),'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps']
import numpy as np,torch,zarr
from mv_probe import build,forward,starts,confusion,overlap,head_state,state_hash
from mv_data import sha,atomic
import torch.nn.functional as F

def read(p):return json.loads(Path(p).read_text())
def csvout(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
 """Enumerate all 212 stores and published roles, without examining test pixels."""
 data=Path('/opt/data/3_117/tanghr/MitoVerse_data');catalog=read(data/'catalog.json');old=read(OLD/'MV-Dev-v1.json')
 assert len(catalog)==212
 overlays={p.stem:read(p) for p in (data/'splits').glob('*.json')}
 roles=collections.defaultdict(list)
 for name,j in overlays.items():
  for v in j.get('volumes',[]):roles[v['id']].append({'benchmark':name,'split':v['split'],'regions':v.get('regions')})
 headtrain={e['id'] for e in old['entries'] if e['supervised_role']=='train'}
 olddev={e['id'] for e in old['entries'] if e['supervised_role']=='dev'}
 # Preserve all previously locked tests and all explicit test overlay assignments.
 reserved=set(old['reserved_test_ids'])|{i for i,rs in roles.items() if any(v['split']=='test' for v in rs)}
 reserved|={i for i,c in catalog.items() if c['dataset_id']=='conrad23' or i.endswith('_test1') or i=='casser20_kasthuri++_test'}
 entries=[];inventory=[];excluded=[]
 for vid,c in sorted(catalog.items()):
  p=data/c['zarr'];attrs=read(p/'.zattrs').get('mitoverse',{});arrays={}
  for n in ['img','mito','mask']:
   if (p/n/'.zarray').exists():arrays[n]=read(p/n/'.zarray')
  shape=c['shape_zyx'];source=c['dataset_id'];sourcefile=attrs.get('source',{}).get('im',attrs.get('source',{}).get('img'))
  acquisition=c.get('imaging',{}).get('sample')
  declared=roles.get(vid,[]);val=any(v['split']=='val' for v in declared)
  reason=None
  if vid in reserved:reason='reserved_test; no test label pixels read'
  elif vid=='casser20_lucchi++':reason='merged benchmark source train/test coordinates unavailable; do not repurpose formal test pixels'
  elif vid in headtrain:reason='used_by_all_current_heads_for_supervised_training; not eligible held-out dev'
  elif arrays.get('img',{}).get('shape')!=shape or arrays.get('mito',{}).get('shape')!=shape:reason='image_label_geometry_unusable'
  elif arrays['img']['dtype']!='|u1' or not np.issubdtype(np.dtype(arrays['mito']['dtype']),np.unsignedinteger):reason='unsupported_label_or_image_dtype'
  elif 'mask' in arrays and arrays['mask']['shape']!=shape:reason='ignore_mask_geometry_unusable'
  # Provenance is a declared annotation contract, not independent re-annotation.
  elif attrs.get('provenance') not in {'instance_proofread','binary_conversion_proofread','ours_proofread','liu26'}:reason='label_semantics_unavailable'
  roi=[[0,n] for n in shape];kind='published_val' if val else 'exploratory_custom_dev'
  evidence='MitoVerse split overlay' if val else 'project exploratory holdout: all usable non-test stores not used by current heads'
  official_case=None
  if vid=='kunduri22_train02':
   kind='official_ME2_val_native_store';official_case='me2-sperm_train02';evidence='Zenodo 20417683 Dataset007 split.json, unique [128,1024,1024] image geometry'
  elif source=='wei20':
   assert shape==[500,4096,4096];roi=[[400,500],[0,4096],[0,4096]]
   kind='official_classic_MitoEM_val_region';evidence='MitoEM Grand Challenge: train0-399, val400-499; stored first500 native slices'
  elif vid=='han24_pos2':
   kind='official_ME2_val_native_store';official_case='me2-mossy_train03';evidence='Zenodo20417683 Dataset004 split.json; unique image geometry [281,1343,1243]'
  elif source=='muller21' and vid in olddev:
   kind='MV1_custom_dev_of_published_TRAIN';evidence='preserved MV1 holdout, not official ME2 validation'
  rawrow={'volume_id':vid,'source_id':source,'acquisition_id':acquisition or '',
   'acquisition_status':'official_sample_name' if acquisition else 'UNVERIFIED',
   'original_image_file':sourcefile or '', 'published_splits':json.dumps(declared),
   'shape_zyx':json.dumps(shape),'spacing_zyx_nm':json.dumps(list(reversed(c['voxel_nm']))),
   'spacing_status':'catalog_declared_UNVERIFIED_Pinky_specific' if source=='turner22' else 'official_ingest_native_grid',
   'annotation_contract':attrs.get('provenance','UNVERIFIED'),'mask_present':'mask' in arrays,
   'mask_policy':'mask>0 is ignore, valid=mask==0; published ingest declares ignore mask' if 'mask' in arrays else 'complete_declared_no_ignore',
   'head_supervised_exposed':vid in headtrain,'B_full_ssl_exposed':True,
   'region_zyx':json.dumps(roi),'dev_kind':kind,'admission':'excluded' if reason else 'usable_dev',
   'reason':reason or 'development usable; acquisition independence not required or claimed'}
  inventory.append(rawrow)
  if reason:excluded.append(rawrow);continue
  entries.append({'id':vid,'volume_id':vid,'source_id':source,'acquisition_id':acquisition,
   'acquisition_status':rawrow['acquisition_status'],'original_image_file':sourcefile,
   'image_path':str(p),'shape_zyx':shape,'region_zyx':roi,'axes':'ZYX',
   'spacing_zyx_nm':list(reversed(c['voxel_nm'])),'spacing_status':rawrow['spacing_status'],
   'dev_kind':kind,'split_evidence':evidence,'official_case':official_case,
   'mask_present':'mask' in arrays,'mask_policy':rawrow['mask_policy'],
   'annotation_contract':attrs.get('provenance'),'ssl_exposed':True,'head_supervised_exposed':False,
   'modality':c['modality'],'declared_instances':attrs.get('n_instances'),
   'metadata_sha256':{str(f.relative_to(p)):sha(f) for f in [p/'.zattrs',*[p/n/'.zarray' for n in arrays]]}})
 m={'name':'MV-2 Full Development Validation','claim':'transductive_frozen_head_development_transfer',
  'inference_only':True,'training_permitted':False,'tta':['identity','horizontal','vertical','horizontal_vertical'],
  'threshold':0.5,'crop_size':256,'stride':128,'normalization':'(uint8/255-0.5)/0.25 grayscale repeatedRGB',
  'metrics':{'dice':'binary valid-region','hd95':'nm, symmetric max directed95th percentile','nsd':'16nm',
    'mask_distance_policy':'only valid surfaces, excluding one-voxel ignore adjacency; no fake ignore boundary'},
  'entries':entries,'reserved_test_ids':sorted(reserved),'head_supervised_train_ids':sorted(headtrain),
  'excluded':excluded,'official_val_missing_native_mapping':[
   {'dataset':'ME2-Beta','case':'me2-beta_train01','status':'head_supervised_exposed','native':'muller21_high_c1','reason':'official val was MV1 head train; exclude held-out ranking'},
   *[{'dataset':'ME2-'+name,'case':'me2-'+name.lower()+'_train02','status':'official val image missing from native stores; native train/test0 not relabeled val'} for name in ['Jurkat','Macro','Podo']],
   {'dataset':'ME2-Pyra','status':'release ZIP has no split.json; do not fabricate ME2 val; classic MitoEM-H val region evaluated separately'}],
  'limitations':['No independent acquisition generalization claim.','Many crops share acquisition; report sample groups separately.',
   'Exploratory native labels may differ from refined ME2 release; this is MitoVerse development, not ME2 protocol replication.',
   'Historical SSL/GT exposure is preserved.','No source lacking legitimate dev is scored zero.']}
 atomic(R/'MV_dev_validation_manifest.json',m);csvout(R/'MV_volume_registry.csv',inventory)
 atomic(R/'evidence/input_hashes.json',{str(p):sha(p) for p in [data/'catalog.json',*[data/'splits'/f'{name}.json' for name in overlays],OLD/'MV-Dev-v1.json',OLD/'MV_frozen_probe_results.csv',OLD/'src/mv_probe.py',OLD/'src/official_primus.py',*[OLD/'runs'/model/f'seed{s}'/'head_last.pth' for model in ['P0','B100k','B150k'] for s in [20261013,20261014,20261015]]]})
 print('MANIFEST_READY',len(entries),dict(collections.Counter(e['source_id'] for e in entries)),flush=True)

def cache(row):
 d=R/'cache'/row['id'];d.mkdir(parents=True,exist_ok=True)
 receipt=d/'QC.json'
 if receipt.exists():
  j=read(receipt);assert j['region_zyx']==row['region_zyx'] and j['metadata_sha256']==row['metadata_sha256'];return d,j
 for f,h in row['metadata_sha256'].items():assert sha(Path(row['image_path'])/f)==h
 group=zarr.open_group(row['image_path'],mode='r');bounds=row['region_zyx'];shape=tuple(b-a for a,b in bounds)
 image=np.lib.format.open_memmap(d/'image.npy',mode='w+',dtype=np.uint8,shape=shape)
 label=np.lib.format.open_memmap(d/'label.npy',mode='w+',dtype=np.uint8,shape=shape)
 valid=np.lib.format.open_memmap(d/'valid.npy',mode='w+',dtype=np.uint8,shape=shape)
 count=total=ignore=ignored_gt=0;hs=[hashlib.sha256() for _ in range(3)]
 for z in range(0,shape[0],8):
  key=(slice(bounds[0][0]+z,min(bounds[0][1],bounds[0][0]+z+8)),slice(*bounds[1]),slice(*bounds[2]))
  a=np.asarray(group['img'][key]);b=np.asarray(group['mito'][key]);v=np.asarray(group['mask'][key])==0 if row['mask_present'] else np.ones(a.shape,bool)
  assert a.shape==b.shape==v.shape and np.isfinite(a).all()
  image[z:z+len(a)]=a;label[z:z+len(a)]=b>0;valid[z:z+len(a)]=v
  count+=int(np.count_nonzero((b>0)&v));total+=int(v.sum());ignore+=int((~v).sum());ignored_gt+=int(np.count_nonzero((b>0)&~v))
  for h,x in zip(hs,[a,b,v]):h.update(x.tobytes())
  if z%64==0:atomic(R/'cache_progress.json',{'volume':row['id'],'slice':z,'slices':shape[0]})
 assert total>0,'No valid pixels; unusable entry'
 for x in [image,label,valid]:x.flush()
 j={'status':'PASS','id':row['id'],'region_zyx':bounds,'shape_zyx':list(shape),'metadata_sha256':row['metadata_sha256'],
  'valid_voxels':total,'foreground_valid_voxels':count,'ignored_voxels':ignore,'foreground_ignored_voxels':ignored_gt,
  'source_pixel_hashes':[h.hexdigest() for h in hs],'test_label_access':False,'mask_policy':row['mask_policy']}
 atomic(receipt,j);print('CACHE_READY',row['id'],total,count,ignore,flush=True);return d,j

def prepare_cache():
 m=read(R/'MV_dev_validation_manifest.json');receipts=[];unusable=[]
 # New small/source volumes first; no dependency on completing all caches.
 for row in sorted(m['entries'],key=lambda e:np.prod([b-a for a,b in e['region_zyx']])):
  try:d,j=cache(row);receipts.append(j)
  except Exception as e:
   unusable.append({'id':row['id'],'reason':type(e).__name__+': '+str(e)});atomic(R/'cache'/row['id']/'UNUSABLE.json',unusable[-1]);print('CACHE_UNUSABLE',unusable[-1],flush=True)
 atomic(R/'evidence/cache_completion.json',{'status':'complete','entries':len(receipts),'unusable':unusable})

@torch.no_grad()
def sliding_tta(net,image):
 H,W=image.shape;acc=np.zeros((H,W),np.float32);weights=np.zeros((H,W),np.uint16)
 coords=[(y,x) for y in starts(H) for x in starts(W)]
 for begin in range(0,len(coords),8):
  tiles=[];items=[]
  for y,x in coords[begin:begin+8]:
   a=np.asarray(image[y:y+256,x:x+256],np.float32);h,w=a.shape
   a=np.pad(a,((0,256-h),(0,256-w)),constant_values=127.5)
   tiles.append(((a/255-.5)/.25).copy());items.append((y,x,h,w))
  batch=torch.from_numpy(np.stack(tiles))[:,None].cuda();p=torch.zeros((len(items),256,256),device='cuda')
  for dims in [(),(-1,),(-2,),(-2,-1)]:
   x=batch.flip(dims) if dims else batch
   q=forward(net,x).softmax(1)[:,1];p+=q.flip(dims) if dims else q
  p=(p/4).cpu().numpy()
  for q,(y,x,h,w) in zip(p,items):acc[y:y+h,x:x+w]+=q[:h,:w];weights[y:y+h,x:x+w]+=1
 assert (weights>0).all();return acc/weights

def infer(model,s):
 out=R/'runs'/model/f'seed{s}';out.mkdir(parents=True,exist_ok=True)
 source=OLD/'runs'/model/f'seed{s}'/'head_last.pth';h=sha(source);saved=torch.load(source,map_location='cpu',weights_only=True)
 assert saved['step']==20000 and saved['metadata']['model']==model and saved['metadata']['seed']==s
 net,spec=build(model,s);net.load_state_dict(saved['head'],strict=False)
 assert state_hash(head_state(net))==state_hash(saved['head']);net.requires_grad_(False).eval();del saved;gc.collect()
 m=read(R/'MV_dev_validation_manifest.json');mh=sha(R/'MV_dev_validation_manifest.json');t0=time.time();allrows=[]
 todo=sorted(m['entries'],key=lambda e:np.prod([b-a for a,b in e['region_zyx']]))
 for row in todo:
  d=R/'cache'/row['id'];result=out/(row['id']+'_overlap.json')
  if result.exists():
   j=read(result);assert j['manifest_sha256']==mh and j['head_sha256']==h;allrows.append(j);continue
  while not (d/'QC.json').exists() and not (d/'UNUSABLE.json').exists():
   if (R/'evidence/cache_completion.json').exists():raise RuntimeError('Cache missing after completion')
   atomic(out/'progress.json',{'model':model,'seed':s,'stage':'waiting_cache','volume':row['id'],'elapsed_s':time.time()-t0});time.sleep(3)
  if (d/'UNUSABLE.json').exists():continue
  im=np.load(d/'image.npy',mmap_mode='r');gt=np.load(d/'label.npy',mmap_mode='r');valid=np.load(d/'valid.npy',mmap_mode='r')
  path=out/(row['id']+'_probability.npy');prob=np.lib.format.open_memmap(path,mode='w+',shape=im.shape,dtype=np.float32)
  c=np.zeros(4,np.int64)
  for z in range(len(im)):
   p=sliding_tta(net,im[z]);assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
   prob[z]=p;v=valid[z].astype(bool);c+=confusion((p>.5)[v],gt[z].astype(bool)[v])
   if z%20==0:atomic(out/'progress.json',{'model':model,'seed':s,'volume':row['id'],'z_completed':z+1,'z_total':len(im),'elapsed_s':time.time()-t0})
  prob.flush();del prob
  j={'model':model,'seed':s,'volume_id':row['id'],'source_id':row['source_id'],'acquisition_id':row['acquisition_id'],
   'acquisition_status':row['acquisition_status'],'dev_kind':row['dev_kind'],'manifest_sha256':mh,'head_sha256':h,
   'encoder_sha256':spec['sha256'],'tta':'four_flip_mean_softmax','head_supervised_exposed':False,'ssl_exposed':True,
   'spacing_zyx_nm':row['spacing_zyx_nm'],'spacing_status':row['spacing_status'],**overlap(c)}
  assert j['tp']+j['fn']==read(d/'QC.json')['foreground_valid_voxels']
  atomic(result,j);allrows.append(j);atomic(out/'overlap_metrics.json',allrows)
  print('INFERRED',model,s,row['id'],j['dice'],flush=True)
 atomic(out/'inference_complete.json',{'status':'complete','model':model,'seed':s,'volumes':len(allrows),'manifest_sha256':mh,'head_sha256':h,'elapsed_s':time.time()-t0})

def distances(pred,gt,valid,spacing):
 from monai.metrics import compute_hausdorff_distance,compute_surface_dice
 # Use MV1's verified shared EDT cache with official MONAI entrypoints on unmasked volumes.
 if valid.all():
  from metric_cache import install
  hd,ns,_=install(compute_hausdorff_distance,compute_surface_dice)
  p=torch.from_numpy(pred.copy())[None,None];g=torch.from_numpy(gt.copy())[None,None]
  return float(hd(p,g,include_background=True,percentile=95,spacing=spacing)[0,0]),float(ns(p,g,include_background=True,class_thresholds=[16],spacing=spacing)[0,0])
 from scipy.ndimage import binary_erosion,distance_transform_edt
 from monai.metrics.utils import get_mask_edges
 # Filter genuine GT/pred surfaces, never create a surface by zeroing an ignore region.
 e1,e2=get_mask_edges(pred,gt,crop=False,always_return_as_numpy=True)
 safe=binary_erosion(valid,border_value=1);e1&=safe;e2&=safe
 if not e1.any() or not e2.any():return math.inf,0.
 d=distance_transform_edt(~e2,sampling=spacing);a=d[e1].copy();del d
 d=distance_transform_edt(~e1,sampling=spacing);b=d[e2].copy();del d
 return max(float(np.percentile(a,95)),float(np.percentile(b,95))),float(((a<=16).sum()+(b<=16).sum())/(len(a)+len(b)))

def score_one(path):
 j=read(path);out=path.with_name(path.name.replace('_overlap.json','_metrics.json'))
 if out.exists():return read(out)
 d=R/'cache'/j['volume_id'];gt=np.load(d/'label.npy',mmap_mode='r').astype(bool);valid=np.load(d/'valid.npy',mmap_mode='r').astype(bool)
 p=np.load(path.with_name(path.name.replace('_overlap.json','_probability.npy')),mmap_mode='r')>.5
 g=gt&valid;pv=p&valid
 if not g.any():hd=0. if not pv.any() else math.inf;ns=1. if not pv.any() else 0.
 elif not pv.any():hd=math.inf;ns=0.
 else:hd,ns=distances(p,gt,valid,j['spacing_zyx_nm'])
 j.update(hd95_nm=hd,nsd_16nm=ns,valid_voxels=int(valid.sum()),gt_foreground_voxels=int(g.sum()),prediction_foreground_voxels=int(pv.sum()))
 # PR and calibration are calculated from the same probabilities, without threshold selection.
 histp=np.zeros(10000,np.int64);histn=np.zeros(10000,np.int64);bs=0.;ececount=np.zeros(20,np.int64);ecesum=np.zeros(20);ecegt=np.zeros(20)
 prob=np.load(path.with_name(path.name.replace('_overlap.json','_probability.npy')),mmap_mode='r')
 for z in range(len(gt)):
  pp=np.asarray(prob[z])[valid[z]];gg=gt[z][valid[z]];bins=np.minimum((pp*10000).astype(int),9999)
  histp+=np.bincount(bins[gg],minlength=10000);histn+=np.bincount(bins[~gg],minlength=10000)
  bs+=float(np.sum((pp.astype(np.float64)-gg)**2));b=np.minimum((pp*20).astype(int),19)
  ececount+=np.bincount(b,minlength=20);ecesum+=np.bincount(b,weights=pp,minlength=20);ecegt+=np.bincount(b,weights=gg,minlength=20)
 hits=np.cumsum(histp[::-1]);draws=hits+np.cumsum(histn[::-1]);prec=np.divide(hits,draws,out=np.ones_like(hits,dtype=float),where=draws>0)
 j['pixel_AP_hist10000']=float(np.sum(prec*histp[::-1])/histp.sum()) if histp.sum() else None
 j['brier']=bs/valid.sum();j['ECE20']=float(np.abs(ecesum-ecegt).sum()/valid.sum())
 j['predicted_foreground_fraction']=float(pv.sum()/valid.sum());atomic(out,j)
 print('SCORED',j['model'],j['seed'],j['volume_id'],hd,ns,flush=True);return j

def score_all():
 """CPU stream: scores new volumes as GPU inference completes; memory bounded."""
 import concurrent.futures,psutil
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
  active={};done=set()
  while True:
   candidates=sorted(R.glob('runs/*/seed*/*_overlap.json'))
   for p in candidates:
    if p in done or p in active:continue
    if p.with_name(p.name.replace('_overlap.json','_metrics.json')).exists():done.add(p);continue
    if len(active)>=2:break
    n=read(R/'cache'/read(p)['volume_id']/'QC.json')['valid_voxels']
    # At most one >=1B-voxel EDT at a time; retain ample memory for GPUs/website.
    large=n>1_000_000_000
    if large and active:continue
    if any(v[1] for v in active.values()):continue
    if psutil.virtual_memory().available<max(32<<30,n*100):continue
    active[p]=(ex.submit(score_one,p),large)
   for p,(f,large) in list(active.items()):
    if f.done():f.result();done.add(p);del active[p]
   if (R/'inference_all_complete.json').exists() and len(done)==len(candidates) and not active:break
   atomic(R/'scoring_progress.json',{'scored':len(done),'inferred':len(candidates),'active':[str(p) for p in active]});time.sleep(5)
 summarize()

def summarize():
 m=read(R/'MV_dev_validation_manifest.json');rows=[read(p) for p in sorted(R.glob('runs/*/seed*/*_metrics.json'))]
 flat=[]
 for r in rows:
  flat.append({k:r.get(k) for k in ['model','seed','volume_id','source_id','acquisition_id','acquisition_status','dev_kind','dice','precision','recall','hd95_nm','nsd_16nm','pixel_AP_hist10000','brier','ECE20','predicted_foreground_fraction','tp','fp','fn','tn','valid_voxels','spacing_status','tta','ssl_exposed','head_supervised_exposed']})
 csvout(R/'MV_full_validation_metrics.csv',flat)
 sources=[];paired=[]
 metrics=['dice','precision','recall','hd95_nm','nsd_16nm']
 def mean(values):return statistics.mean([x for x in values if x is not None]) if values else None
 for s in sorted({r['source_id'] for r in rows}):
  for model in ['P0','B100k','B150k']:
   rs=[r for r in rows if r['source_id']==s and r['model']==model];seedvalues={}
   groups={r['acquisition_id'] or 'UNVERIFIED:'+r['volume_id'] for r in rs}
   row={'source_id':s,'model':model,'n_volumes':len({r['volume_id'] for r in rs}),'n_seeds':len({r['seed'] for r in rs}),'n_confirmed_acquisition_names':len({r['acquisition_id'] for r in rs if r['acquisition_id']}),'n_unverified_acquisition_volumes':len({r['volume_id'] for r in rs if not r['acquisition_id']})}
   for metric in metrics:
    vals=[];groupvals=[]
    for seed in sorted({r['seed'] for r in rs}):
     rr=[r for r in rs if r['seed']==seed];vals.append(mean([r[metric] for r in rr]))
     gv=[mean([r[metric] for r in rr if (r['acquisition_id'] or 'UNVERIFIED:'+r['volume_id'])==g]) for g in groups];groupvals.append(mean(gv))
    row[metric+'_mean']=mean(vals);row[metric+'_sd_seeds']=statistics.stdev(vals) if len(vals)>1 and all(math.isfinite(v) for v in vals) else None
    row[metric+'_known_acquisition_group_mean']=mean(groupvals)
   sources.append(row)
 for r in rows:
  if r['model']!='B100k':continue
  p=next(x for x in rows if x['model']=='P0' and x['seed']==r['seed'] and x['volume_id']==r['volume_id'])
  paired.append({'volume_id':r['volume_id'],'source_id':r['source_id'],'seed':r['seed'],**{k+'_B100_minus_P0':r[k]-p[k] for k in metrics}})
 csvout(R/'MV_source_comparison.csv',sources);csvout(R/'MV_paired_deltas.csv',paired)
 coverage=[]
 for s in sorted({r['source_id'] for r in read_csv(R/'MV_volume_registry.csv')}):
  seen=[r for r in rows if r['source_id']==s];included=[r for r in m['entries'] if r['source_id']==s]
  coverage.append({'source_id':s,'dev_entries':len(included),'evaluated_volumes':len({r['volume_id'] for r in seen}),
   'metric_rows':len(seen),'status':'evaluated' if seen else 'not_evaluated','reason':'; '.join(sorted({r['reason'] for r in m['excluded'] if r['source_id']==s})) if not seen else 'official and/or explicitly custom development; never independent crop acquisitions'})
 csvout(R/'MV_source_coverage.csv',coverage)
 atomic(R/'metrics_all_complete.json',{'status':'complete','rows':len(rows),'volumes':len({r['volume_id'] for r in rows}),'sources':len({r['source_id'] for r in rows}),'tta':'four_flip','test_labels_accessed':False})
 print('ALL_METRICS_COMPLETE',len(rows),flush=True)
def read_csv(p):return list(csv.DictReader(p.open()))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','cache','infer','score','summarize']);a.add_argument('--model');a.add_argument('--seed',type=int);v=a.parse_args()
 if v.mode=='prepare':prepare()
 elif v.mode=='cache':prepare_cache()
 elif v.mode=='infer':infer(v.model,v.seed)
 elif v.mode=='score':score_all()
 else:summarize()
