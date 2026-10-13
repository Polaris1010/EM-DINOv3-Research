"""Existing frozen Primus inference on the full usable development union. No training."""
from pathlib import Path
import sys,json,hashlib,collections,csv,math,time,os,gc,argparse,statistics
R=Path(__file__).resolve().parents[1]
OLD=R.parent/'mv_generalist_stage1_20261013'
sys.path[:0]=[str(OLD/'src'),'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps']
import numpy as np,torch,zarr
from mv_probe import build,sliding,confusion,overlap,head_state,state_hash
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
  'inference_only':True,'training_permitted':False,'tta':'OFF','tile_batch_size':16,
  'inference_function':'unchanged mv_generalist_stage1_20261013/src/mv_probe.py::sliding',
  'reuse_existing_MV1_dev_results':sorted(olddev),
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
 oldrow=next((v for v in read(OLD/'MV-Dev-v1.json')['entries'] if v['id']==row['id'] and v['supervised_role']=='dev'),None)
 if oldrow:
  assert oldrow['region_zyx']==row['region_zyx'] and oldrow['metadata_sha256']==row['metadata_sha256']
  j=read(OLD/'cache'/row['id']/'QC.json')
  j={**j,'region_zyx':row['region_zyx'],'metadata_sha256':row['metadata_sha256'],
   'valid_voxels':j['total_voxels'],'foreground_valid_voxels':j['foreground_voxels'],'ignored_voxels':0,
   'cache_origin':'MV1_verified_cache_and_metrics_reused_without_new_inference'}
  atomic(receipt,j);return d,j
 prior=R.parent/'mv_dev_v2_expansion_20261013'/'cache'/row['id']
 if (prior/'QC.json').exists():
  j=read(prior/'QC.json')
  assert j['region_zyx']==row['region_zyx'] and j['metadata_sha256']==row['metadata_sha256']
  for name in ['image.npy','label.npy','valid.npy']:
   target=d/name
   if not target.exists():target.symlink_to(prior/name)
  j={**j,'cache_origin':'reused identical TTA-independent source/GT cache; NO four-flip predictions reused'}
  atomic(receipt,j);return d,j
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
  if row['id'] in m['reuse_existing_MV1_dev_results']:
   previous=next(v for v in read_csv(OLD/'runs'/model/f'seed{s}'/'per_volume_metrics.csv') if v['volume_id']==row['id'])
   oldmeta=read(OLD/'runs'/model/f'seed{s}'/'metrics.json')
   assert oldmeta['manifest_sha256']==sha(OLD/'MV-Dev-v1.json') and oldmeta['validation_tile_batch_size']==16
   oldrow=next(v for v in read(OLD/'MV-Dev-v1.json')['entries'] if v['id']==row['id'])
   assert oldrow['region_zyx']==row['region_zyx'] and oldrow['metadata_sha256']==row['metadata_sha256']
   numbers={k:float(previous[k]) for k in ['dice','precision','recall','hd95_nm','nsd_16nm']}
   counts={k:int(previous[k]) for k in ['tp','fp','fn','tn']}
   j={'model':model,'seed':s,'volume_id':row['id'],'source_id':row['source_id'],'acquisition_id':row['acquisition_id'],
    'acquisition_status':row['acquisition_status'],'dev_kind':row['dev_kind'],'manifest_sha256':mh,'head_sha256':h,
    'encoder_sha256':spec['sha256'],'tta':'OFF','head_supervised_exposed':False,'ssl_exposed':True,
    'spacing_zyx_nm':row['spacing_zyx_nm'],'spacing_status':row['spacing_status'],**numbers,**counts,
    'foreground_empty_prediction':previous['empty_prediction']=='True','valid_voxels':sum(counts.values()),
    'gt_foreground_voxels':counts['tp']+counts['fn'],'prediction_foreground_voxels':counts['tp']+counts['fp'],
    'result_origin':'MV1_direct_reuse_NO_RERUN','old_metric_csv_sha256':sha(OLD/'runs'/model/f'seed{s}'/'per_volume_metrics.csv')}
   atomic(result,j);atomic(result.with_name(result.name.replace('_overlap.json','_metrics.json')),j)
   allrows.append(j);print('REUSED_MV1_NO_RERUN',model,s,row['id'],flush=True);continue
  while not (d/'QC.json').exists() and not (d/'UNUSABLE.json').exists():
   if (R/'evidence/cache_completion.json').exists():raise RuntimeError('Cache missing after completion')
   atomic(out/'progress.json',{'model':model,'seed':s,'stage':'waiting_cache','volume':row['id'],'elapsed_s':time.time()-t0});time.sleep(3)
  if (d/'UNUSABLE.json').exists():continue
  im=np.load(d/'image.npy',mmap_mode='r');gt=np.load(d/'label.npy',mmap_mode='r');valid=np.load(d/'valid.npy',mmap_mode='r')
  path=out/(row['id']+'_prediction.npy');pred=np.lib.format.open_memmap(path,mode='w+',shape=im.shape,dtype=np.uint8)
  c=np.zeros(4,np.int64)
  for z in range(len(im)):
   p=sliding(net,im[z],tile_batch_size=16)
   pred[z]=p;v=valid[z].astype(bool);c+=confusion(p[v],gt[z].astype(bool)[v])
   if z%20==0:atomic(out/'progress.json',{'model':model,'seed':s,'volume':row['id'],'z_completed':z+1,'z_total':len(im),'elapsed_s':time.time()-t0})
  pred.flush();del pred
  j={'model':model,'seed':s,'volume_id':row['id'],'source_id':row['source_id'],'acquisition_id':row['acquisition_id'],
   'acquisition_status':row['acquisition_status'],'dev_kind':row['dev_kind'],'manifest_sha256':mh,'head_sha256':h,
   'encoder_sha256':spec['sha256'],'tta':'OFF','head_supervised_exposed':False,'ssl_exposed':True,'result_origin':'new_region_MV1_sliding_unchanged',
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
  from large_hd95 import compatibility
  with compatibility():
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
 p=np.load(path.with_name(path.name.replace('_overlap.json','_prediction.npy')),mmap_mode='r')>0
 g=gt&valid;pv=p&valid
 if not g.any():hd=0. if not pv.any() else math.inf;ns=1. if not pv.any() else 0.
 elif not pv.any():hd=math.inf;ns=0.
 else:hd,ns=distances(p,gt,valid,j['spacing_zyx_nm'])
 j.update(hd95_nm=hd,nsd_16nm=ns,valid_voxels=int(valid.sum()),gt_foreground_voxels=int(g.sum()),prediction_foreground_voxels=int(pv.sum()))
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
 strata()
 summarize()

def summarize():
 m=read(R/'MV_dev_validation_manifest.json');rows=[read(p) for p in sorted(R.glob('runs/*/seed*/*_metrics.json'))]
 flat=[]
 for r in rows:
  flat.append({k:r.get(k) for k in ['model','seed','volume_id','source_id','acquisition_id','acquisition_status','dev_kind','dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction','tp','fp','fn','tn','valid_voxels','spacing_status','tta','ssl_exposed','head_supervised_exposed','result_origin']})
 csvout(R/'MV_full_validation_metrics.csv',flat)
 sources=[];paired=[]
 metrics=['dice','precision','recall','hd95_nm','nsd_16nm']
 def mean(values):
  values=[x for x in values if x is not None]
  return statistics.mean(values) if values else None
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
    row[metric+'_mean']=mean(vals);row[metric+'_sd_seeds']=statistics.stdev(vals) if len(vals)>1 and all(v is not None and math.isfinite(v) for v in vals) else None
    row[metric+'_known_acquisition_group_mean']=mean(groupvals)
   sources.append(row)
 for r in rows:
  if r['model']!='B100k':continue
  p=next(x for x in rows if x['model']=='P0' and x['seed']==r['seed'] and x['volume_id']==r['volume_id'])
  paired.append({'volume_id':r['volume_id'],'source_id':r['source_id'],'seed':r['seed'],**{k+'_B100_minus_P0':r[k]-p[k] if r[k] is not None and p[k] is not None else None for k in metrics}})
 csvout(R/'MV_source_comparison.csv',sources);csvout(R/'MV_paired_deltas.csv',paired)
 coverage=[]
 for s in sorted({r['source_id'] for r in read_csv(R/'MV_volume_registry.csv')}):
  seen=[r for r in rows if r['source_id']==s];included=[r for r in m['entries'] if r['source_id']==s]
  coverage.append({'source_id':s,'dev_entries':len(included),'evaluated_volumes':len({r['volume_id'] for r in seen}),
   'metric_rows':len(seen),'status':'evaluated' if seen else 'not_evaluated','reason':'; '.join(sorted({r['reason'] for r in m['excluded'] if r['source_id']==s})) if not seen else 'official and/or explicitly custom development; never independent crop acquisitions'})
 csvout(R/'MV_source_coverage.csv',coverage)
 atomic(R/'metrics_all_complete.json',{'status':'complete','rows':len(rows),'volumes':len({r['volume_id'] for r in rows}),'sources':len({r['source_id'] for r in rows}),'tta':'OFF','test_labels_accessed':False})
 write_reports(m,rows,sources,paired,coverage)
 print('ALL_METRICS_COMPLETE',len(rows),flush=True)
def read_csv(p):return list(csv.DictReader(p.open()))

def strata():
 """First-seed GT-instance voxel recall, full admitted regions, no instance-detection claim."""
 rows=[];m=read(R/'MV_dev_validation_manifest.json');countsout=[]
 for row in m['entries']:
  paths=[R/'runs'/model/'seed20261013'/(row['id']+'_metrics.json') for model in ['P0','B100k','B150k']]
  if not all(p.exists() for p in paths):continue
  bounds=row['region_zyx'];group=zarr.open_group(row['image_path'],mode='r');c={};hits={model:{} for model in ['P0','B100k','B150k']};touch=set();preds={}
  for model in hits:
   root=OLD if row['id'] in m['reuse_existing_MV1_dev_results'] else R
   preds[model]=np.load(root/'runs'/model/'seed20261013'/(row['id']+'_prediction.npy'),mmap_mode='r')
  for zi,z in enumerate(range(*bounds[0])):
   key=(z,slice(*bounds[1]),slice(*bounds[2]));ids=np.asarray(group['mito'][key]);v=np.asarray(group['mask'][key])==0 if row['mask_present'] else np.ones(ids.shape,bool)
   n=int(ids.max())+1;cc=np.bincount(ids[v].ravel(),minlength=n)
   for k in np.flatnonzero(cc[1:])+1:c[int(k)]=c.get(int(k),0)+int(cc[k])
   if zi in {0,bounds[0][1]-bounds[0][0]-1}:touch.update(map(int,np.unique(ids)))
   touch.update(map(int,np.unique(np.concatenate([ids[0],ids[-1],ids[:,0],ids[:,-1],ids[~v]]))))
   for model,p in preds.items():
    h=np.bincount(ids[(p[zi]>0)&v].ravel(),minlength=n)
    for k in np.flatnonzero(h[1:])+1:hits[model][int(k)]=hits[model].get(int(k),0)+int(h[k])
  whole=[n for k,n in c.items() if k not in touch];cuts=np.quantile(whole,[1/3,2/3]) if len(whole)>=3 else None
  for k,n in c.items():
   size='ROI_or_ignore_truncated' if k in touch else 'insufficient_nontruncated_instances' if cuts is None else 'small' if n<=cuts[0] else 'medium' if n<=cuts[1] else 'large'
   for model in hits:rows.append({'volume_id':row['id'],'source_id':row['source_id'],'acquisition_id':row['acquisition_id'],
    'model':model,'seed':20261013,'instance_id':k,'gt_valid_voxels':n,'size_nm3':float(n*np.prod(row['spacing_zyx_nm'])),
    'size_bin_within_volume':size,'touches_ROI_or_ignore':k in touch,'voxel_recall':hits[model].get(k,0)/n,
    'physical_spacing_status':row['spacing_status']})
  countsout.append({'volume_id':row['id'],'instances_with_valid_GT':len(c),'nontruncated_instances':len(whole)});print('STRATA_DONE',row['id'],len(c),flush=True)
 csvout(R/'MV_object_strata_seed1.csv',rows);atomic(R/'evidence/object_strata.json',{'status':'complete','seed':20261013,'rows':len(rows),'volumes':countsout,'definition':'per-GT-instance voxel recall, not instance detection recall'})

def write_reports(m,rows,sources,paired,coverage):
 def fmt(v):return 'NA' if v is None else f'{v:.6f}'
 def avg(vals):
  vals=[v for v in vals if v is not None]
  return statistics.mean(vals) if vals else None
 lines=['# MV-2 全开发验证结果','',f"完成 {len({r['source_id'] for r in rows})} 来源、{len({r['volume_id'] for r in rows})} 个volume/region，3 encoder×3 decoder seeds。",
  '','现有20k Primus heads直接推理；encoder/head均未训练。TTA=OFF，直接调用MV1原始sliding函数，256窗口、128步长、batch16、均值softmax、阈值0.5。MV1原4个开发体积的36条指标直接复用，没有重跑。',
  '','## 每来源均值 ± decoder seed SD','',
  '|来源|模型|volume数|已确认sample名称数|Dice|HD95 nm|NSD@16nm|Precision|Recall|',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in sources:
  values=[fmt(r[k+'_mean'])+' ± '+fmt(r[k+'_sd_seeds']) for k in ['dice','hd95_nm','nsd_16nm','precision','recall']]
  lines.append(f"|{r['source_id']}|{r['model']}|{r['n_volumes']}|{r['n_confirmed_acquisition_names']}|"+'|'.join(values)+'|')
 lines+=['','## B100k−P0 同体积同seed配对','', '|来源|ΔDice均值|ΔRecall均值|ΔNSD均值|三个seed来源Dice差值|','|---|---:|---:|---:|---|']
 decisions=[]
 for s in sorted({r['source_id'] for r in paired}):
  rs=[r for r in paired if r['source_id']==s];seedmeans=[statistics.mean(r['dice_B100_minus_P0'] for r in rs if r['seed']==seed) for seed in [20261013,20261014,20261015]]
  delta=statistics.mean(seedmeans);decisions.append({'source':s,'delta':delta,'seedmeans':seedmeans})
  lines.append('|'+s+'|'+fmt(delta)+'|'+fmt(avg([r['recall_B100_minus_P0'] for r in rs]))+'|'+fmt(avg([r['nsd_16nm_B100_minus_P0'] for r in rs]))+'|'+', '.join(map(fmt,seedmeans))+'|')
 lines+=['','## 漏检与小目标','',
  '本轮主指标是语义体素Recall，不是实例检测Recall。各来源低Recall、较高Precision的组合提示漏检或保守预测，但单凭二值预测不能区分校准偏移和表征可分性下降。',
  '本轮MV_object_strata_seed1.csv覆盖所有完成推理的开发区域：按原始GT实例统计有效体素Recall及物理体积，ROI/ignore截断对象单列。仅首seed分层，不等同于三个seed稳定的实例检测结论。小/中/大按每体积非截断实例的三分位定义，样本少于3个时不硬分箱；不是跨来源统一尺寸阈值。',
  '','## 解释范围','',
  '- 全量B_full无标签图像暴露：transductive开发评测；不是独立unseen-image测试。',
  '- 官方val与显式exploratory_custom_dev在manifest/CSV中逐项分开。本轮不是统一官方全局validation split，也不是全MitoVerse监督训练后的generalist模型。',
  '- 多crop可共享OpenOrganelle sample/acquisition；CSV另报sample group均值。未知采集体不伪造独立数量；三个decoder seeds也不是三个生物采集体。',
  '- 只使用已有模型监督训练之外的合法开发数据；正式test保留。ME2-Beta官方val对应high_c1，已是MV1 head训练体积，因此不纳入held-out分数。',
  '- ME2-Jurkat/Macro/Podo的正式train02图像目前未能对应本地native stores；不把本地test1或train/test0偷换为官方val。Pyra正式ZIP没有split.json；另按classic MitoEM的400:500区域评测。',
  '- 物理间距使用MitoVerse原生ZYX nm；Turner/Pinky细化间距仍为catalog声明值，相关物理指标带UNVERIFIED标记。正式ME2发布部分NIfTI头与native spacing不同，未混用。',
  '- 有ignore mask时，仅统计有效像素；距离指标使用真实表面并排除ignore邻接，避免制造mask边界。空前景/空预测保留并明确HD95 inf，不能静默剔除。',
  '- 未覆盖来源及具体原因见MV_source_coverage.csv、manifest.excluded。未覆盖不记零分。',
  '','## 未纳入项','']
 for c in coverage:
  if c['status']!='evaluated':lines.append(f"- {c['source_id']}: {c['reason']}")
 if (R/'evidence/cache_completion.json').exists():
  for u in read(R/'evidence/cache_completion.json')['unusable']:lines.append('- '+u['id']+': '+u['reason'])
 if (R/'MV_object_strata_seed1.csv').exists():
  objects=read_csv(R/'MV_object_strata_seed1.csv');lines+=['','## 首seed小目标体素Recall','', '|来源|非截断小目标数|P0|B100k|B150k|','|---|---:|---:|---:|---:|']
  for s in sorted({r['source_id'] for r in objects}):
   selected=[r for r in objects if r['source_id']==s and r['size_bin_within_volume']=='small']
   vals=[avg([float(r['voxel_recall']) for r in selected if r['model']==model]) for model in ['P0','B100k','B150k']]
   n=sum(r['model']=='P0' for r in selected);lines.append('|'+s+'|'+str(n)+'|'+'|'.join(map(fmt,vals))+'|')
 (R/'MV_failure_analysis.md').write_text('\n'.join(lines)+'\n')
 positive=[d['source'] for d in decisions if d['delta']>0];negative=[d['source'] for d in decisions if d['delta']<0]
 if positive and negative:
  decision='转向来源相关迁移差异这一研究假设；下一阶段先用更广泛监督数据训练统一generalist Primus head，并与MitoNet比较。'
  interpretation='本轮直接测得正负来源差异，支持优先研究跨来源负迁移假设；尚未排除当前head只在四个体积训练、监督优化或校准带来的影响，不能宣称encoder被证明受损。'
 elif positive:
  decision='继续发展EM表征优势；下一阶段训练覆盖更多来源的统一generalist Primus head，再与MitoNet比较。'
  interpretation='本轮来源均值支持B100k更广泛的收益；仍是transductive开发结果，不能代替独立测试。'
 else:
  decision='暂不延长vanilla SSL；下一阶段以P0和B100k匹配的generalist Primus head对照检验现有EM适配的实际价值。'
  interpretation='本轮没有测得跨来源普遍的正收益，不足以支持继续加训练步数。'
 text=['# MV2 研究决策','', '**决定：** '+decision,'','**直接证据：**',
  f"- 完成{len(coverage)}个来源的覆盖清单，实际评测{len(decisions)}个来源。",
  '- B100k−P0来源平均Dice为正：'+(', '.join(positive) or '无')+'；为负：'+(', '.join(negative) or '无')+'。',
  '- 同体积、同decoder seed、同MV1无TTA协议；完整配对数据见MV_paired_deltas.csv。','',interpretation,'',
  '**下一行动：** 固定原版Primus，扩展合规监督训练数据，仅先比较P0和B100k；使用共同训练预算与匹配种子，随后部署MitoNet同图像语义对照。',
  '', '**本轮结束：** 不自动启动下一轮训练，不增加SSL模块。Gram/HRA、Structure-aware iBOT或跨来源鲁棒SSL只在generalist对照明确主要差异后选一项短程验证。']
 (R/'MV2_research_decision.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','cache','infer','score','summarize']);a.add_argument('--model');a.add_argument('--seed',type=int);v=a.parse_args()
 if v.mode=='prepare':prepare()
 elif v.mode=='cache':prepare_cache()
 elif v.mode=='infer':infer(v.model,v.seed)
 elif v.mode=='score':score_all()
 else:summarize()
