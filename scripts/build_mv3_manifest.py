"""Version the authorized MV-3 split from existing metadata; never read pixels."""
import argparse,collections,copy,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build(snapshot):
 catalog=snapshot['catalog'];mv2=snapshot['mv2_manifest'];old=snapshot['old_manifest']
 reserved=set(mv2['reserved_test_ids']);known={r['volume_id']:r for r in mv2['entries']}
 known.update({r['id']:r for r in old['entries']})
 official={r['volume_id'] for r in mv2['entries'] if r['dev_kind'] in {'published_val','official_ME2_val_native_store'}}
 official.add('muller21_high_c1')
 oo=sorted({c.get('imaging',{}).get('sample') for c in catalog.values() if c['dataset_id']=='openorganelle'})
 oo_holdout=set(sorted(oo,key=lambda g:hashlib.sha256(('MV-Generalist-v1:'+g).encode()).hexdigest())[:4])
 rows=[];excluded=[]
 for vid,c in sorted(catalog.items()):
  arrays=snapshot['stores'][vid]['arrays'];attrs=snapshot['stores'][vid]['attrs']['mitoverse'];shape=c['shape_zyx']
  reason=None
  if vid in reserved:reason='reserved_official_test_no_pixel_access'
  elif vid=='casser20_lucchi++':reason='merged_official_train_test_coordinates_unavailable'
  elif vid=='wei20_mitoEM-H':reason='ME2-Pyra_heldout_test_region_coordinates_unavailable; exclude_asset_without_new_audit'
  elif arrays.get('img',{}).get('shape')!=shape or arrays.get('mito',{}).get('shape')!=shape:reason='known_image_label_geometry_unusable'
  elif attrs.get('provenance') not in {'instance_proofread','binary_conversion_proofread','ours_proofread','liu26'}:reason='annotation_contract_unavailable'
  if reason:
   excluded.append({'volume_id':vid,'source_id':c['dataset_id'],'reason':reason});continue
  source=c['dataset_id'];roi=[[0,n] for n in shape];role='train';kind='reassigned_exploratory_MV2_to_train'
  if vid in official:role='dev';kind='preserved_official_validation'
  elif source=='wei20':
   # Existing classic MitoEM train0:400 / validation400:500; official test500:1000 is absent.
   for split,bounds in [('train',[0,400]),('dev',[400,500])]:
    row=base_row(vid,c,attrs,arrays,snapshot,known);row.update(id=vid+'__'+split,supervised_role=split,dev_kind='official_classic_MitoEM_'+split+'_region',region_zyx=[bounds,*roi[1:]])
    rows.append(row)
   continue
  elif source=='openorganelle' and c.get('imaging',{}).get('sample') in oo_holdout:role='dev';kind='exploratory_acquisition_group_holdout'
  elif source=='turner22' and 'microns1_' in vid:role='dev';kind='exploratory_microns1_group_holdout_all_pinky_variants_train'
  elif vid in {'haberl18_vol1','mekuc20_fib1-4-3-0','muller21_low_c2'}:role='dev';kind='exploratory_volume_holdout'
  elif source in {'casser20','xiao18'}:
   cut=math.floor(shape[0]*.8)
   for split,bounds in [('train',[0,cut]),('dev',[cut,shape[0]])]:
    row=base_row(vid,c,attrs,arrays,snapshot,known);row.update(id=vid+'__'+split,supervised_role=split,dev_kind='exploratory_disjoint_Z_same_acquisition_'+split,region_zyx=[bounds,*roi[1:]])
    rows.append(row)
   continue
  row=base_row(vid,c,attrs,arrays,snapshot,known);row.update(id=vid,supervised_role=role,dev_kind=kind,region_zyx=roi);rows.append(row)
 m={'name':'MV-Generalist-v1','version':1,'status':'LOCKED_BEFORE_SMOKE','claim':'transductive_multisource_supervised_development','training_permitted':True,'unseen_acquisition_claim_permitted':False,'reserved_test_ids':sorted(reserved),'entries':rows,'excluded':excluded,'openorganelle_heldout_acquisition_names':sorted(oo_holdout),'selection_rule':'fixed_step40000_for_both_encoders; 10000/20000 monitoring_only; never independently extend or select best step','known_official_val_missing_native_mapping':mv2['official_val_missing_native_mapping'][1:],'limitations':['Casser and Xiao use disjoint Z of one acquisition; no independent acquisition claim.','Other source acquisition identity limitations are inherited, not re-audited.','OpenOrganelle split is by existing declared sample group, not crop count.','Han nonzero terminal masks are valid; outside including positive labels is ignored. MV2 mask convention differs; historical numbers remain intact.','Wei-H and Lucchi mixed formal test regions are excluded because their existing coordinate mapping is unavailable. Wei-R supplies Wei supervised train and official classic validation.','Turner nm spacing remains inherited UNVERIFIED; no new resampling.','MV2 and MV3 supervision differ; evaluate improvements on matching new heldout regions using stored predictions only.']}
 validate(m);return m

def base_row(vid,c,attrs,arrays,snapshot,known):
 source=c['dataset_id'];path='/opt/data/3_117/tanghr/MitoVerse_data/'+c['zarr'];old=known.get(vid,{})
 return {'volume_id':vid,'source_id':source,'image_path':path,'shape_zyx':c['shape_zyx'],'axes':'ZYX','spacing_zyx_nm':list(reversed(c['voxel_nm'])),'spacing_status':'catalog_declared_UNVERIFIED_Pinky_specific' if source=='turner22' else 'official_ingest_native_grid','acquisition_id':c.get('imaging',{}).get('sample'),'acquisition_status':'official_sample_name' if c.get('imaging',{}).get('sample') else 'UNVERIFIED','original_image_file':attrs.get('source',{}).get('im',attrs.get('source',{}).get('img')),'annotation_contract':attrs['provenance'],'mask_present':'mask' in arrays,'valid_mask_policy':'mask!=0; outside ignore_index=-1' if source=='han24' else 'mask==0; nonzero ignore_index=-1' if 'mask' in arrays else 'complete_declared_no_ignore','mask_policy_evidence':'existing_20260917_mask_policy_v2_and_Han24_repair' if source=='han24' else 'existing_MV2_ingest_contract','metadata_sha256':old.get('metadata_sha256',{}),'ssl_exposed':True,'historical_MV1_supervised_exposed':vid in {r['id'] for r in snapshot['old_manifest']['entries'] if r['supervised_role']=='train'}}

def validate(m):
 rows=m['entries'];assert len({r['id'] for r in rows})==len(rows)
 assert {r['source_id'] for r in rows if r['supervised_role']=='train'}=={r['source_id'] for r in rows if r['supervised_role']=='dev'}
 assert len({r['source_id'] for r in rows if r['supervised_role']=='train'})==12
 for a in rows:
  assert a['volume_id'] not in m['reserved_test_ids'] and a['axes']=='ZYX'
  assert a['supervised_role'] in {'train','dev'}
  assert all(0<=x<y<=n for (x,y),n in zip(a['region_zyx'],a['shape_zyx']))
  if a['source_id']=='han24':
   if m.get('version',1)==1:assert a['valid_mask_policy'].startswith('mask!=0')
   elif m['version']==2:assert a['mask_usage']=='NONE' and a['valid_mask_policy']=='full_GT_no_annotation_mask_USER_AUTHORIZED'
   else:raise ValueError('Unknown Han protocol version')
  for b in rows:
   if a['supervised_role']!=b['supervised_role'] and a['image_path']==b['image_path']:
    assert any(y<=u or v<=x for (x,y),(u,v) in zip(a['region_zyx'],b['region_zyx'])),'Train/dev voxel overlap'
 assert all(r['supervised_role']=='dev' for r in rows if r['dev_kind']=='preserved_official_validation')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('metadata',type=Path);a=ap.parse_args()
 m=build(json.loads(a.metadata.read_text(encoding='utf-8')));folder=ROOT/'configs/segmentation/MV-Generalist-v1'
 folder.mkdir(parents=True,exist_ok=True);p=folder/'manifest.json';p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
 counts={s:dict(collections.Counter(r['supervised_role'] for r in m['entries'] if r['source_id']==s)) for s in sorted({r['source_id'] for r in m['entries']})}
 print(json.dumps({'manifest':str(p),'regions':len(m['entries']),'sources':counts,'openorganelle_groups_heldout':m['openorganelle_heldout_acquisition_names']}))
