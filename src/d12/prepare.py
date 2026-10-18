"""Prepare image-only random-access CEM index and shared held-out MitoVerse pool."""
import collections,hashlib,os,sqlite3,time,zipfile
from common import R,M3,ZIP,WEIGHT,WEIGHT_SHA,read,atomic,sha,subtract_z,cem_exclusion

def main():
 t=time.time();assert sha(WEIGHT)==WEIGHT_SHA
 original=read(R/'inputs/pretrain_all_mitoverse.json');mv=read(M3/'MV-Generalist.json')
 excluded=set(mv['reserved_test_ids'])|{r['volume_id'] for r in mv['excluded']}
 held={};acq={}
 for row in mv['entries']:
  if row.get('acquisition_id'):acq[row['volume_id']]=row['acquisition_id']
  if row['supervised_role']=='dev':held.setdefault(row['volume_id'],[]).append(row['region_zyx'])
 kept=[];removed=[]
 for row in original['volumes']:
  vid=row['volume_id'];why=[]
  if vid in excluded:why.append('existing_official_test_validation_or_unmapped_mixed_asset')
  if row['source_dataset']=='conrad23' or 'lucchi' in vid or 'openorganelle_jrc_fly' in vid:why.append('MitoNet_benchmark_or_related_acquisition')
  intervals=[[0,row['shape'][0]]]
  for box in held.get(vid,[]):
   assert box[1:]==[[0,row['shape'][1]],[0,row['shape'][2]]],'Unexpected partial XY holdout; do not silently leak'
   intervals=subtract_z(intervals,box[0])
  if not intervals:why.append('entire_asset_in_fixed_MV3_dev')
  if why:removed.append({'volume_id':vid,'reasons':why});continue
  a=acq.get(vid)
  if a is None and row['source_dataset']=='openorganelle':a=vid.split('openorganelle_',1)[1].split('_recon',1)[0]
  kept.append({**row,'allowed_z':intervals,'acquisition_sampling_key':a or vid,'acquisition_status':'declared_sample_group' if a else 'unknown_use_volume_proxy','legal_slices':sum(b-a for a,b in intervals)})
 assert kept
 bysource=collections.defaultdict(list)
 for row in kept:bysource[row['source_dataset']].append(row)
 groups={s:{a:[v['volume_id'] for v in vv if v['acquisition_sampling_key']==a] for a in sorted({v['acquisition_sampling_key'] for v in vv})} for s,vv in sorted(bysource.items())}
 mp={'training_role':'ssl_image_holdout_v1','volumes':kept,'excluded':removed,'volume_count':len(kept),'legal_slices':sum(v['legal_slices'] for v in kept),'source_count':len(groups),'sampling_groups':groups,'heldout_regions':held,'original_manifest_sha256':sha(R/'inputs/pretrain_all_mitoverse.json'),'downstream_manifest_sha256':sha(M3/'MV-Generalist.json'),'limits':['Disjoint image regions do not imply acquisition independence','Unknown CEM collection aliases are not certified disjoint','B100 historical SSL exposure and100k budget differ']}
 atomic(R/'mitoverse_pool.json',mp)
 db=R/'cem_index.sqlite';assert not db.exists(),'Refuse to replace an existing data index'
 con=sqlite3.connect(db);con.execute('PRAGMA journal_mode=OFF');con.execute('PRAGMA synchronous=OFF');con.execute('CREATE TABLE images(id INTEGER PRIMARY KEY,collection TEXT,path TEXT,offset INTEGER,compressed INTEGER,bytes INTEGER,crc INTEGER,method INTEGER)')
 counter=collections.Counter();skipped=collections.Counter();group_ranges={};dirs=0;n=0;chain=hashlib.sha256();batch=[]
 with zipfile.ZipFile(ZIP) as archive:
  for info in archive.infolist():
   if info.is_dir():dirs+=1;continue
   assert info.filename.endswith(('.tif','.tiff')) and not (info.flag_bits&1) and info.compress_type in [0,8]
   parts=info.filename.split('/');assert len(parts)==3 and parts[0]=='cem1.5M' and '..' not in parts
   group=parts[1]
   if cem_exclusion(group):skipped[group]+=1;continue
   counter[group]+=1;bounds=group_ranges.setdefault(group,[n,n]);bounds[1]=n
   item=(n,group,info.filename,info.header_offset,info.compress_size,info.file_size,info.CRC,info.compress_type)
   chain.update(repr(item).encode());batch.append(item);n+=1
   if len(batch)==5000:con.executemany('INSERT INTO images VALUES(?,?,?,?,?,?,?,?)',batch);batch=[]
  if batch:con.executemany('INSERT INTO images VALUES(?,?,?,?,?,?,?,?)',batch)
  assert n+sum(skipped.values())==1592753
 con.commit();con.close()
 total=sum(v**.5 for v in counter.values());cg=[]
 for group,count in sorted(counter.items()):
  first,last=group_ranges[group];assert last-first+1==count,'Collection IDs not contiguous'
  cg.append({'name':group,'images':count,'first_id':first,'last_id':last,'probability':count**.5/total,'acquisition_status':'original_collection_not_certified_acquisition'})
 cp={'official_accession':'EMPIAR-11035','archive':str(ZIP),'archive_bytes':ZIP.stat().st_size,'archive_mtime_ns':ZIP.stat().st_mtime_ns,'official_image_count':1592753,'original_nonempty_collections':len(counter)+len(skipped),'zip_directory_entries':dirs,'images':n,'groups':cg,'excluded_collections':[{'name':k,'images':v,'reason':cem_exclusion(k)} for k,v in sorted(skipped.items())],'index':str(db),'index_sha256':sha(db),'record_chain_sha256':chain.hexdigest(),'label_reads':False,'archive_integrity':'ZIP central directory and per-selected-member CRC32; no publisher archive SHA available','benchmark_holdout':'Author methods explicitly exclude benchmark and related images; no new full pixel deduplication','remaining_overlap':'Unmapped hashed collections vs MV3 dev/test remain UNVERIFIED; not strict unseen-acquisition proof'}
 atomic(R/'cem_pool.json',cp);atomic(R/'prepare_complete.json',{'status':'PASS','MitoVerse_volumes':len(kept),'MitoVerse_sources':len(groups),'MitoVerse_slices':mp['legal_slices'],'CEM_images':n,'CEM_collections':len(cg),'seconds':time.time()-t,'gt_read_for_ssl':False})
 print('PREPARE_PASS',read(R/'prepare_complete.json'),flush=True)
if __name__=='__main__':main()
