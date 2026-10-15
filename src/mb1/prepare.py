"""Pin existing checkpoints, official native TIFF geometry, and bounded source links."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import tifffile,zarr
from common import R,M3,M4,M5,OLD,read,sha,atomic,config,Planes
NAMES=['c_elegans','fly_brain','glycolytic_muscle','hela_cell','lucchi_pp','salivary_gland']
SHAPES=[[256,256,256],[256,255,255],[302,383,765],[256,256,256],[165,768,1024],[1260,1081,1200]]
def header(path):
 with tifffile.TiffFile(path) as tf:
  s=tf.series[0];p=tf.pages[0]
  return {'shape':list(s.shape),'axes':s.axes,'dtype':str(s.dtype),'pages':len(tf.pages),'photometric':str(p.photometric),'description':p.description,'resolution_unit':str(p.tags['ResolutionUnit'].value) if 'ResolutionUnit' in p.tags else None,'x_resolution':p.tags['XResolution'].value if 'XResolution' in p.tags else None,'y_resolution':p.tags['YResolution'].value if 'YResolution' in p.tags else None,'imagej_metadata':tf.imagej_metadata}
def tem_spacing(h):
 # Only an explicit physical ImageJ unit supplies calibration. TIFF default DPI is not EM pixel size.
 meta=h.get('imagej_metadata') or {};unit=str(meta.get('unit','')).lower();factor={'nm':1.,'nanometer':1.,'nanometre':1.,'um':1000.,'micron':1000.,'µm':1000.}.get(unit)
 if factor is None:return None,'UNVERIFIED; no explicit physical per-image calibration'
 out=[]
 for key in ['y_resolution','x_resolution']:
  v=h.get(key)
  if not v or v[0]<=0:return None,'UNVERIFIED; incomplete explicit ImageJ calibration'
  out.append(factor*v[1]/v[0])
 return out,'TIFF explicit ImageJ physical unit and XY resolution'
def main():
 c=config();index=read(R/'assets/official_download_index.json');existing=read(R/'evidence/existing_official_files.json');assert existing['matched']==212
 lookup={f['relative']:f for f in existing['files']};checkpoints={}
 for run in c['runs']:
  for path,expected in [(run['checkpoint_path'],run['checkpoint_sha256'])]+([(run['encoder_path'],run['encoder_sha256'])] if run['family']=='dinov3' else []):
   if path in checkpoints:continue
   p=Path(path);digest=sha(p);assert digest==expected,(path,digest,expected);checkpoints[path]={'sha256':digest,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns};print('CHECKPOINT_VERIFIED',p.name,flush=True)
 atomic(R/'evidence/checkpoint_verification.json',{'status':'PASS','assets':checkpoints,'checkpoint_writes':False})
 datasets=[];entries=[];metadata=read(R/'assets/empiar_entry.json')['EMPIAR-10982']['imagesets'];files=[]
 for bid,name in enumerate(NAMES,1):
  pair=[]
  for kind in ['em','mito']:
   rel=f'mito_benchmarks/{name}/{name}_{kind}.tif';f=lookup[rel];p=Path(f['existing_path']);assert p.stat().st_size==f['bytes'];h=header(p)
   assert h['shape']==SHAPES[bid-1],(name,h['shape']);assert np.issubdtype(np.dtype(h['dtype']),np.integer)
   if kind=='em':assert h['dtype']=='uint8'
   j={**f,'header':h,'sha256':sha(p)};files.append(j);pair.append(j);print('RAW_TIFF_VERIFIED',name,kind,flush=True)
  spacing=[float(metadata[bid-1]['pixel_height'])/10,float(metadata[bid-1]['pixel_width'])/10]
  row={'id':name,'dataset_id':bid,'dataset':name,'type':'volume','shape_zyx':pair[0]['header']['shape'],'axis_interpretation':'native TIFF frame order Z,Y,X; no transpose','image_path':pair[0]['existing_path'],'label_path':pair[1]['existing_path'],'image_file_sha256':pair[0]['sha256'],'gt_file_sha256':pair[1]['sha256'],'spacing_yx_nm':spacing,'spacing_status':'official EMPIAR pixel_height/width Angstrom divided by10; user table corroborates','z_spacing_nm':None,'gt_semantic_rule':'mito>0; all native pixels scored','planes':SHAPES[bid-1][0]};entries.append(row);datasets.append({'id':bid,'name':name,'type':'volume','image_count':1,'planes':row['planes'],'spacing_yx_nm':spacing})
 image_files=sorted(f for f in lookup if f.startswith('tem_benchmark/images/'));label_files=sorted(f for f in lookup if f.startswith('tem_benchmark/masks/'));assert len(image_files)==len(label_files)==100
 for rel in image_files:
  mate=rel.replace('/images/','/masks/');assert mate in lookup
  aa,bb=lookup[rel],lookup[mate];ha=header(aa['existing_path']);hb=header(bb['existing_path']);assert ha['shape']==hb['shape'] and len(ha['shape'])==2 and ha['dtype']=='uint8'
  spacing,status=tem_spacing(ha);stem=Path(rel).stem
  pair=[]
  for f,h in [(aa,ha),(bb,hb)]:
   assert Path(f['existing_path']).stat().st_size==f['bytes'];j={**f,'header':h,'sha256':sha(f['existing_path'])};files.append(j);pair.append(j)
  entries.append({'id':'TEM_'+stem,'image_id':stem,'dataset_id':7,'dataset':'TEM-100','type':'image','shape_zyx':[1,*ha['shape']],'axis_interpretation':'native Y,X; singleton Z for storage only','image_path':aa['existing_path'],'label_path':bb['existing_path'],'image_file_sha256':pair[0]['sha256'],'gt_file_sha256':pair[1]['sha256'],'spacing_yx_nm':spacing,'spacing_status':status,'z_spacing_nm':None,'gt_semantic_rule':'mito>0; all native pixels scored','planes':1})
 datasets.append({'id':7,'name':'TEM-100','type':'independent_images','image_count':100,'planes':100,'calibrated_images':sum(e['dataset_id']==7 and e['spacing_yx_nm'] is not None for e in entries)})
 assert len(entries)==106 and sum(e['planes'] for e in entries)==2595
 atomic(R/'MB1_benchmark_manifest.json',{'status':'LOCKED','doi':c['data_doi'],'datasets':datasets,'entries':entries,'original_tiffs_unchanged':True,'reused_official_file_count':212,'expected_model_runs':8,'expected_per_image_metric_rows':20760,'source':'official EMPIAR directory, metadata and torch-em get_benchmark_paths','TEM_metadata_csv':'EMPIAR description mentions CSV; public TEM directory and existing directory contain only images/masks, no CSV found; no invented calibration'})
 atomic(R/'evidence/original_file_index.json',{'files':files});print('MANIFEST_READY',106,2595,flush=True)
 # Small source/asset cross-check, not a search over transforms or all SSL images.
 train=read(M3/'MV-Generalist.json')['entries'];ssl=read(R/'assets/ssl_manifest.json')['volumes'];overlap=[]
 for row in entries[:6]:
  candidate='casser20_lucchi++' if row['id']=='lucchi_pp' else 'conrad23_'+row['id'];match=next((v for v in ssl if v['volume_id']==candidate),None)
  item={'dataset':row['dataset'],'candidate_volume_id':candidate,'ssl_listed':match is not None,'supervised_exact_volume_id_present':any(e['volume_id']==candidate and e['supervised_role']=='train' for e in train),'supervised_note':'candidate source volume excluded/reserved in MV3; no direct candidate volume used by any of six heads or MitoNet-MV','exact_full_pixel_match':'UNVERIFIED; bounded three-plane asset check only'}
  if match and match['shape']==row['shape_zyx']:
   source=zarr.open_group(match['local_path'],mode='r')['img'];raw=Planes(row['image_path'],row['shape_zyx']);zs=sorted({0,row['planes']//2,row['planes']-1});checks=[{'z':z,'equal':bool(np.array_equal(raw[z],np.asarray(source[z])))} for z in zs];raw.close();item.update(status='CONFIRMED_SOURCE_AND_SAMPLED_IMAGE_OVERLAP' if all(x['equal'] for x in checks) else 'CONFIRMED_SOURCE_ASSOCIATION_PIXEL_MAPPING_UNCONFIRMED',sampled_planes=checks,source_shape=match['shape'])
  elif match:item.update(status='CONFIRMED_SOURCE_ASSOCIATION_EXACT_REGION_UNCONFIRMED',source_shape=match['shape'],note='Lucchi++ related source but different geometry; no resizing or spatial search performed')
  else:item.update(status='UNCONFIRMED')
  overlap.append(item)
 overlap.append({'dataset':'TEM-100','status':'NO_OVERLAP_FOUND_IN_AVAILABLE_REGISTRY','note':'no TEM benchmark filenames/source found in available212-volume registry; absence of names does not prove absence of image exposure; no exhaustive pixel matching','ssl_exact_image_exposure':'UNVERIFIED','supervised_exact_image_exposure':'UNVERIFIED'})
 atomic(R/'evidence/overlap_source_check.json',{'rows':overlap,'not_strict_external_test':True,'official_mitonet_upstream_image_exposure':'CEM-MitoLab and CEM1.5M; exact image-level overlap not independently reaudited'})
 atomic(R/'prepare_complete.json',{'status':'PASS','manifest_sha256':sha(R/'MB1_benchmark_manifest.json'),'checkpoint_assets':len(checkpoints),'files':212,'planes':2595,'gpu_used':False})
if __name__=='__main__':main()
