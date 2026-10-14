"""CPU-only image provenance checks; no inference, tuning or data mutation."""
import ast,csv,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import zarr
BASE=Path('/opt/data/3_117/tanghr');R=BASE/'codex/mv4c_input_provenance_20261015';M3=BASE/'codex/mv_generalist_v2_han_fullgt_20261014';M4B=BASE/'codex/mv4b_semantic_panoptic_20261014'
IDS=['guay21_vol1','muller21_high_c1']
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,j):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(j,indent=2)+'\n')
def stats(hist):
 x=np.arange(len(hist));n=int(hist.sum());mean=float((x*hist).sum()/n);std=float(np.sqrt(((x-mean)**2*hist).sum()/n));cdf=np.cumsum(hist);nz=np.flatnonzero(hist)
 return {'n':n,'min':int(nz[0]),'max':int(nz[-1]),'mean':mean,'std':std,'quantiles':{str(q):int(np.searchsorted(cdf,q/100*(n-1)+1)) for q in [0.5,1,25,50,75,99,99.5]},'zero_fraction':int(hist[0])/n,'max_value_fraction':int(hist[-1])/n}
def load_original(path):
 if path.suffix.lower() in ['.h5','.hdf5']:
  import h5py
  with h5py.File(path,'r') as f:return np.asarray(f[list(f.keys())[0]]),{'reader':'h5 first dataset as official ingest','dataset':list(f.keys())[0]}
 import tifffile
 with tifffile.TiffFile(path) as t:
  page=t.pages[0];info={'reader':'tifffile.imread-equivalent, matching official ingest; no geometric transform','tiff_axes':t.series[0].axes,'photometric':str(page.photometric.name),'orientation_tag':str(page.tags['Orientation'].value) if 'Orientation' in page.tags else 'not set (default top-left)','samples_per_pixel':page.samplesperpixel}
  return t.asarray(),info
def protected(config):
 files=[M3/'MV-Generalist.json',M3/'MV3_protocol.json',M3/'MV3_per_region_metrics.csv',M4B/'MV4B_protocol.json',M4B/'MV4B_summary.json',M4B/'STATUS.json']
 arrays=[]
 for region in IDS:
  c=read(M3/'cache'/region/'CACHE.json');files += [M3/'cache'/region/'CACHE.json',M4B/'runs/Official-MitoNet-Semantic-v1'/(region+'_inference.json')]
  arrays += [Path(c['paths'][k]) for k in ['image','label']]+list((M4B/'runs/Official-MitoNet-Semantic-v1').glob(region+'*.npy'))
 return {'hashes':{str(p):sha(p) for p in files},'arrays_stat':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in arrays}}
def original_mapping(raw):
 # Execute only the unchanged official to_uint8 definition, never writer/main.
 source=BASE/'mitoverse/mitoverse/scripts/ingest.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='to_uint8');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
 mapped,norm=ns['to_uint8'](raw);return mapped,norm,sha(source)
def figures(row,group,image,gt,prob,raw,normalization,result):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 planes=row['figure_planes'];out=R/'figures';out.mkdir(exist_ok=True);records=[]
 for z in planes:
  fig,axes=plt.subplots(2,3,figsize=(15,10))
  if raw is None:
   axes[0,0].set_facecolor('#e9e9e9');axes[0,0].text(.5,.5,'Pre-ingest original unavailable\nNot reconstructed from Zarr\nNot a Zarr-export TIFF',ha='center',va='center',transform=axes[0,0].transAxes);axes[0,0].set_title('Original before ingest: UNVERIFIED');axes[0,0].set_xticks([]);axes[0,0].set_yticks([])
  else:
   axes[0,0].imshow(raw[z],cmap='gray_r' if result['original'].get('photometric')=='MINISWHITE' else 'gray',interpolation='nearest',vmin=float(raw.min()),vmax=float(raw.max()));axes[0,0].set_title('Verified pre-ingest original (linear display)');axes[0,0].axis('off')
  zi=np.asarray(group['img'][z]);mask=gt[z]>0
  axes[0,1].imshow(zi,cmap='gray',vmin=0,vmax=255);axes[0,1].set_title('Current Zarr: uint8, fixed 0..255');axes[0,1].axis('off')
  axes[0,2].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[0,2].set_title('Existing GT at identical native Z/XY');axes[0,2].axis('off')
  pp=np.asarray(prob[z]);a=axes[1,0].imshow(pp,cmap='magma',vmin=0,vmax=1);axes[1,0].set_title('Existing raw semantic probability (0..1)');axes[1,0].axis('off');fig.colorbar(a,ax=axes[1,0],fraction=.045,pad=.02)
  axes[1,1].imshow(zi,cmap='gray',vmin=0,vmax=255)
  if mask.any():axes[1,1].contour(mask,levels=[.5],colors=['#00d5ff'],linewidths=.65)
  axes[1,1].set_title('Zarr + GT outline (no alignment fitting)');axes[1,1].axis('off')
  axes[1,2].bar(np.arange(256),result['image_histogram'],width=1,color='#555555');axes[1,2].set_title('Whole-region Zarr intensity histogram');axes[1,2].set_xlabel('stored uint8 intensity');axes[1,2].set_ylabel('pixel count')
  fig.suptitle(f'{row["region_id"]}; native Z={z}; native XY unchanged\n{normalization}; probability reused, no inference or parameter changes')
  fig.subplots_adjust(left=.03,right=.97,top=.88,bottom=.07,wspace=.20,hspace=.28)
  p=out/(row['region_id']+f'_z{z:04d}.png');fig.savefig(p,dpi=110);plt.close(fig)
  records.append({'region_id':row['region_id'],'native_z':z,'file':str(p),'sha256':sha(p),'original_panel_verified':raw is not None})
 return records
def audit():
 c=read(R/'MV4C_protocol.json');assert [r['region_id'] for r in c['regions']]==IDS and c['inference_performed'] is False
 os.environ['CUDA_VISIBLE_DEVICES']='';before=protected(c);dump(R/'evidence/protected_before.json',before);catalog=read(BASE/'MitoVerse_data/catalog.json');results=[];allfig=[]
 for row in c['regions']:
  region=row['region_id'];meta=catalog[region];store=BASE/'MitoVerse_data'/meta['zarr'];group=zarr.open_group(str(store),mode='r')
  cache=read(M3/'cache'/region/'CACHE.json');assert cache['offsets']==cache['region_zyx']==[[0,n] for n in cache['shape_zyx']]
  im=np.load(cache['paths']['image'],mmap_mode='r');gt=np.load(cache['paths']['label'],mmap_mode='r');probpath=M4B/'runs/Official-MitoNet-Semantic-v1'/(region+'_probability.npy');prob=np.load(probpath,mmap_mode='r');receipt=read(probpath.with_name(region+'_inference.json'))
  shape=tuple(cache['shape_zyx']);assert im.shape==gt.shape==prob.shape==group['img'].shape==group['mito'].shape==shape
  hist=np.zeros(256,np.int64);fghist=np.zeros(256,np.int64);mismatch=gt_mismatch=0;hh=hashlib.sha256();zh=hashlib.sha256();gh=hashlib.sha256();ph=hashlib.sha256();blocks=[]
  for start in range(0,shape[0],64):
   stop=min(shape[0],start+64);zi=np.asarray(group['img'][start:stop]);zg=np.asarray(group['mito'][start:stop])>0
   ca=np.asarray(im[start:stop]);cg=np.asarray(gt[start:stop])>0
   diff=int(np.count_nonzero(zi!=ca));gd=int(np.count_nonzero(zg!=cg));mismatch+=diff;gt_mismatch+=gd
   hist+=np.bincount(zi.ravel(),minlength=256);fghist+=np.bincount(zi[cg].ravel(),minlength=256)
   hh.update(ca.tobytes(order='C'));zh.update(zi.tobytes(order='C'));gh.update(cg.astype(np.uint8).tobytes(order='C'));ph.update(np.asarray(prob[start:stop]).tobytes(order='C'))
   blocks.append({'z_start':start,'z_stop':stop,'image_cache_mismatch':diff,'gt_cache_mismatch':gd})
   dump(R/'STATUS.json',{'status':'running','phase':'zarr_cache_alignment','region':region,'z_complete':stop,'z_total':shape[0],'gpu_used':False})
  assert hh.hexdigest()==receipt['input_image_sha256'],'Current cached input differs from probability-generation input'
  available=Path(row['original_image']) if row.get('original_image') else Path(meta['source']['im']);raw=None;original={'path':str(available),'exists':available.exists(),'status':'UNVERIFIED_MISSING_PRE_INGEST_FILE'}
  if available.exists():
   raw,info=load_original(available);original.update(info,dtype=str(raw.dtype),shape=list(raw.shape),file_sha256=sha(available));assert raw.shape==shape,'Original must match native shape; no searched registration/cropping allowed'
   mapped,norm,source_sha=original_mapping(raw);neq=0;err=0;sumraw=sumzi=sumraw2=sumzi2=sumprod=0.
   for start in range(0,shape[0],64):
    end=min(shape[0],start+64);zi=np.asarray(group['img'][start:end]);ra=raw[start:end];ma=mapped[start:end];neq+=int(np.count_nonzero(ma!=zi));err=max(err,int(np.abs(ma.astype(np.int16)-zi.astype(np.int16)).max()))
    a=ra.astype(np.float64).ravel();b=zi.astype(np.float64).ravel();sumraw+=float(a.sum());sumzi+=float(b.sum());sumraw2+=float(np.dot(a,a));sumzi2+=float(np.dot(b,b));sumprod+=float(np.dot(a,b))
   n=int(np.prod(shape));den=((sumraw2-sumraw**2/n)*(sumzi2-sumzi**2/n))**.5;corr=(sumprod-sumraw*sumzi/n)/den
   original.update(status='VERIFIED_EXACT_INGEST_MAPPING' if neq==0 else 'OBSERVED_MAPPING_MISMATCH',reproduced_normalization=norm,ingest_source_sha256=source_sha,normalization_mismatch_voxels=neq,max_abs_mapped_error=err,raw_to_zarr_pearson_same_coordinates=corr,spatial_mapping='identity ZYX; no flip/transpose/registration search',grayscale_direction='preserved by positive official mapping' if neq==0 and corr>0 else 'UNRESOLVED')
   del mapped
  result={'region_id':region,'normalization_recorded':meta['normalization'],'original':original,'shape_zyx':list(shape),'zarr_path':str(store),'zattrs':dict(group.attrs),'zarr_dtype':str(group['img'].dtype),'zarr_axes':'ZYX','metadata_voxel_xyz_nm':meta['voxel_nm'],
   'zarr_to_input_cache_mismatch_voxels':mismatch,'zarr_GT_foreground_to_cache_mismatch_voxels':gt_mismatch,'blocks':blocks,'cached_input_sha256':hh.hexdigest(),'zarr_image_values_sha256':zh.hexdigest(),'binary_GT_values_sha256':gh.hexdigest(),'probability_values_sha256':ph.hexdigest(),'probability_original_input_sha256':receipt['input_image_sha256'],
   'probability_path':str(probpath),'probability_dtype':str(prob.dtype),'probability_reused':True,'prediction_registered_region_zyx':receipt['native_region_zyx'],'image_histogram':hist.tolist(),'image_stats':stats(hist),'GT_foreground_intensity_stats':stats(fghist),
   'official_model_normalization':'(uint8/255 -0.57571)/0.12765; same direction; no new model call','normalized_mean':(stats(hist)['mean']/255-.57571)/.12765,'normalized_std':stats(hist)['std']/255/.12765,'raw_to_zarr_status':original['status'],'cache_alignment_verified':mismatch==0 and gt_mismatch==0,
   'original_GT_alignment':'UNVERIFIED without pre-ingest label file; current Zarr GT and cache correspondence independently measured','inference_performed':False,'threshold_tuned':False,'scale_searched':False}
  allfig+=figures(row,group,im,gt,prob,raw,meta['normalization'],result);results.append(result);dump(R/(region+'_audit.json'),result)
 after=protected(c);assert after==before;dump(R/'evidence/protected_after.json',after);dump(R/'MV4C_figures.json',allfig)
 dump(R/'MV4C_summary.json',{'status':'complete' if all(r['original']['exists'] for r in results) else 'AVAILABLE_CHECKS_COMPLETE_ORIGINALS_UNAVAILABLE','regions':results,'historical_inputs_unchanged':True,'gpu_used':False,'inference_performed':False})
 dump(R/'STATUS.json',{'status':'complete_available_checks','regions':2,'figures':len(allfig),'original_images_verified':sum(r['original']['status']=='VERIFIED_EXACT_INGEST_MAPPING' for r in results),'historical_inputs_unchanged':True,'gpu_used':False,'new_inference':False,'new_training':False})
 print('MV4C_CPU_AUDIT_COMPLETE',[(r['region_id'],r['original']['status'],r['zarr_to_input_cache_mismatch_voxels']) for r in results],flush=True)
if __name__=='__main__':audit()
