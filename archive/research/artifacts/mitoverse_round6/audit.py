"""Read-only input audit; writes only into this independent round6 directory."""
import os,sys,json,csv,hashlib,shutil,subprocess,re,time
from pathlib import Path
from settings import *
sys.path.insert(0,str(ROOT/'codex/mitotopo_20260908_round2/deps'))
import numpy as np,zarr
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(8*1024**2),b''):h.update(a)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,default=str))
def table(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 with (BASE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 archive=BASE/'archive';archive.mkdir(exist_ok=True);refs=[]
 def copy(p,key):
  if not p.exists():return
  dst=archive/key;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
  refs.append(dict(source=str(p),archived=str(dst.relative_to(BASE)),sha256=sha(dst)))
 for p in R4.glob('*.py'):copy(p,'round4/'+p.name)
 for name in ['split_mitoverse_probe_v1.json','pretraining_cache_evidence.json','schedule_audit.json','train_foreground_coordinates.json']:
  copy(R4/name,'round4/'+name)
 for seed in range(3):
  for name in [f'shared_head_init_seed{seed}.pth',f'train_patch_schedule_seed{seed}.jsonl']:copy(R4/name,'round4/'+name)
 for p in R5.iterdir():
  if p.is_file() and p.suffix in ['.py','.json','.csv','.md']:copy(p,'round5/'+p.name)
 for j in sorted((R5/'jobs').glob('v4_b16*')):
  for p in j.iterdir():
   if p.is_file() and p.suffix in ['.json','.yaml','.yml']:copy(p,'round5/jobs/'+j.name+'/'+p.name)
 for seed in range(3):copy(R4/f'jobs/evaluate_P1_seed{seed}_N2/metrics.json',f'round4/P1_seed{seed}_metrics.json')
 for p in [DATA/'catalog.json',DATA/'README.md',DATA/'.cache/huggingface/download/catalog.json.metadata']:
  copy(p,'data/'+p.name)
 for p in (DATA/'splits').glob('*.json'):copy(p,'data/splits/'+p.name)
 for p in [ROOT/'mitoverse/mitoverse/mitoverse/io.py',ROOT/'mitoverse/mitoverse/scripts/ingest.py',ROOT/'mitoverse/mitoverse/catalog/datasets.yaml',ROOT/'DINOv3/pretrain_data/prepare_jrc_new.py',ROOT/'DINOv3/med_checkpoints_continual_v2/config.yaml']:
  copy(p,'source/'+p.name)
 observed=sha(P1);save(BASE/'weight_hashes.json',dict(path=str(P1),expected=P1_SHA,observed=observed,match=observed==P1_SHA))
 assert observed==P1_SHA,'P1 hash mismatch: dependent tasks must stop'
 repos={}
 for name,p in [('dinov3',ROOT/'DINOv3/dinov3'),('meddinov3',ROOT/'Med_DINOv3/MedDINOv3'),('mitoverse',ROOT/'mitoverse/mitoverse')]:
  def git(*args):
   q=subprocess.run(['git','-C',str(p),*args],text=True,capture_output=True);return dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
  repos[name]=dict(path=str(p),commit=git('rev-parse','HEAD'),status=git('status','--porcelain'))
  (archive/f'{name}_working_tree.patch').write_text(git('diff','HEAD')['stdout'])
 save(BASE/'repository_state.json',repos)
 versions=subprocess.run([str(PYTHON),'-m','pip','freeze'],capture_output=True,text=True);(archive/'dependencies.txt').write_text(versions.stdout)
 cat=json.loads((DATA/'catalog.json').read_text());split=json.loads((R4/'split_mitoverse_probe_v1.json').read_text());manifest=[];quarantine=[];overlap=[];exposures=[]
 jrc={'jrc_cos7-1a','jrc_ctl-id8-1','jrc_jurkat-1','jrc_macrophage-2','jrc_mus-kidney','jrc_mus-liver','jrc_sum159-1'}
 aliases={'casser20':'Casser/Lucchi++/Kasthuri++','wei20':'MitoEM/Wei/ME2-Pyra','han24':'Han/MitoLE/ME2-Mossy','muller21':'BetaSeg/Muller/ME2-Beta','kunduri22':'Kunduri/ME2-Sperm','jiang25':'Jiang/ME2-Stem','openorganelle':'OpenOrganelle/COSEM/CellMap/JRC/MitoEM2 overlay','turner22':'MICrONS/Pinky/Turner','conrad23':'MitoNet/EMPIAR-10982'}
 for vid,c in cat.items():
  p=DATA/c['zarr'];a=json.loads((p/'.zattrs').read_text()).get('mitoverse',{});source=c['dataset_id'];old=split['volumes'][vid];sp=old['split'];sp='legacy_dev_test' if sp.startswith('test') else sp
  copy(p/'.zattrs',f'data/zattrs/{vid}.json')
  imeta=json.loads((p/'img/.zarray').read_text());lmeta=json.loads((p/'mito/.zarray').read_text());assert imeta['shape']==lmeta['shape']
  sample=a.get('imaging',{}).get('sample') or c.get('imaging',{}).get('sample') or ''
  # An explicit original JRC scan is identifiable. Do not turn an exported ROI filename into a specimen ID.
  orig=sample;specimen=sample
  if source=='openorganelle' and not orig:
   match=re.match(r'(jrc_.+?)(?:_recon-\d+)?_(?:crop\d+|test\d+)$',vid.removeprefix('openorganelle_'))
   if match:orig=match.group(1);specimen=orig
  identity='explicit imaging.sample / JRC scan identifier' if orig else 'original specimen/scan mapping absent in local metadata'
  p1=old.get('p1_em_seen','uncertain');p1='uncertain' if p1 in ['no','unknown',''] else p1
  if source=='wei20':p1='same volume (MitoEM cache); exact ROI uncertain'
  if 'lucchi' in vid:p1='same dataset; original volume/ROI uncertain'
  r5='same original scan; ROI uncertain' if orig in jrc else 'no named R5 JRC scan match; independence not proven'
  r=dict(volume_id=vid,local_path=str(p),dataset_alias=aliases.get(source,source),original_dataset_id=source,specimen_id=specimen,original_volume_id=orig,original_url=json.dumps(a.get('imaging',c.get('imaging',{}))),roi_origin='unknown',roi_shape=json.dumps(imeta['shape']),axes='ZYX',units='nm',voxel_size_zyx=json.dumps(list(reversed(c['voxel_nm']))),split=sp,original_split=old['split'],p1_historical_exposure=p1,round5_exposure=r5,proposed_ssl_exposure='none pending identity gate',evidence=str(p/'.zattrs')+'; local catalog; archived split',confidence='scan identity known; ROI origin unknown' if orig else 'unresolved original identity',mask_present=(p/'mask/.zarray').exists(),catalog_provenance=c.get('provenance'),actual_provenance=a.get('provenance'),identity_evidence=identity)
  manifest.append(r)
  if sp=='train' and not orig:quarantine.append(dict(volume_id=vid,reason='No reliable original specimen AND original scan mapping; local source/ROI names alone cannot establish split independence',scope='new SSL only; diagnostic patches may use annotated train data'))
  exposures.extend([dict(volume_id=vid,checkpoint=k,exposure=v) for k,v in [('P1',p1),('R5_C_G',r5),('round6','not trained')]])
  if r['catalog_provenance']!=r['actual_provenance']:overlap.append(dict(a=vid,b='catalog vs zattrs',relation='metadata provenance drift',evidence=r['evidence'],action='archive both; actual label arrays authoritative'))
  if orig in jrc:overlap.append(dict(a=vid,b='R5 JRC '+orig,relation='same original scan; ROI intersection unresolved',evidence=r['evidence'],action='exclude from mutually exclusive new MitoVerse SSL pool'))
 overlay=json.loads((DATA/'splits/mitoem2.0.json').read_text())
 for row in overlay['volumes']:
  overlap.append(dict(a=row['id'],b=row['me2'],relation='same underlying zarr; overlay not independent data',evidence=str(DATA/'splits/mitoem2.0.json'),action='retain existing exploratory source split; no official ME2 benchmark claim'))
 # Local CMCC/CMCT conversion assets: filename evidence is a candidate linkage, never proof of independence.
 cm=[]
 for dataset in ['Dataset103_CMCC_Mito','Dataset131_CMCT']:
  root=ROOT/'Med_DINOv3/nnUNet_raw'/dataset
  for subset in ['imagesTr','imagesTs']:
   for p in sorted((root/subset).glob('*.nii.gz')):cm.append(dict(dataset=dataset,split=subset,file=str(p),basename=p.name))
  train={p.name for p in (root/'imagesTr').glob('*.nii.gz')};test={p.name for p in (root/'imagesTs').glob('*.nii.gz')}
  for name in sorted(train&test):
   x=root/'imagesTr'/name;y=root/'imagesTs'/name;equal=sha(x)==sha(y)
   overlap.append(dict(a=str(x),b=str(y),relation='byte-identical image across Tr/Ts' if equal else 'same original ROI filename across Tr/Ts; differing files inconclusive',evidence='SHA256 comparison',action='legacy conversion assets only; not assumed to be actual prior eval split; quarantine new SSL candidates'))
 table('cmcc_cmct_asset_index.csv',cm)
 for alias,reason in [('IsoOrg-1K','No local original-scan manifest linked to these assets; public OpenOrganelle origin is not local identity evidence'),('anonymous P1 OpenOrganelle crops','Coordinate-named crops lack original-scan mapping; P1 historical exposure remains uncertain')]:
  quarantine.append(dict(volume_id=alias,reason=reason,scope='new SSL identity audit'))
 table('data_manifest.csv',manifest);table('overlap_audit.csv',overlap);table('exposure_by_checkpoint.csv',exposures);table('quarantine.csv',quarantine)
 # No fabricated identity resolution. Old JRC scan names can be proposed, but are not declared safe against unidentified legacy specimens.
 save(BASE/'ssl_mitoverse_allowlist.json',dict(status='blocked',volumes=[],reason='All 31 currently assigned train volumes lack reliable original specimen/scan identifiers; source grouping alone is insufficient for requested identity gate.'))
 save(BASE/'ssl_jrc_allowlist.json',dict(status='blocked_pending_linkage',volumes=[],candidate_original_scans=sorted(jrc),reason='CMCC/CMCT and anonymous historical crops require scan-level linkage; counterpart TM pool unavailable.'))
 mask=[]
 for r in manifest:
  if not r['mask_present']:continue
  g=zarr.open_group(r['local_path'],mode='r');a=g['mask'];b=g['mito'];counts={};fgout=fgin=0;total=valid=0
  # Suspected mask bug only: five bounded slices, not a whole-library scan.
  for z in np.linspace(0,a.shape[0]-1,5,dtype=int):
   m=np.asarray(a[z]);l=np.asarray(b[z]);u,n=np.unique(m,return_counts=True)
   for v,c in zip(u,n):counts[str(int(v))]=counts.get(str(int(v)),0)+int(c)
   fgin+=int(((l>0)&(m!=0)).sum());fgout+=int(((l>0)&(m==0)).sum());total+=m.size;valid+=int((m!=0).sum())
  mask.append(dict(volume_id=r['volume_id'],values=counts,foreground_in_nonzero=fgin,foreground_in_zero=fgout,nonzero_fraction=valid/total,sampled_slices=5,meaning='nonzero fits annotated-cell region; polarity not explicitly documented',disposition='block formal han24 supervision until authoritative polarity confirmed; diagnostics use other sources'))
 save(BASE/'mask_audit.json',mask)
 save(BASE/'archive_index.json',refs)
 (BASE/'split_audit.md').write_text('# Round6 data gate\n\nLocal HF revision: '+(DATA/'.cache/huggingface/download/catalog.json.metadata').read_text().splitlines()[0]+'\n\nExisting split retained: 31 train / 8 validation / 16 legacy_dev_test / 157 excluded. No difficult volume moved. The six test sources are previously consulted development tests.\n\n**New SSL blocked:** all current target training volumes have incomplete original specimen/scan provenance. Grouping by publication is not proof of independent specimens. No fabricated identifiers or SSL allowlisting. JRC named scans overlap excluded MitoVerse OpenOrganelle assets; ROI origins are absent, so overlap extent is unresolved. MitoEM2 is an overlay, not independent data. CMCC/CMCT conversion assets are indexed separately and must not be assumed to define prior evaluation splits.\n\n**Mask issue:** han24 carries region-valued masks. Sampled positive labels lie within nonzero mask, but the importer only says “ignore mask” without polarity. Formal han24 supervision remains blocked pending authoritative semantics; fixed-patch diagnostics use unmasked training sources.\n\nHistorical P1 exposure is retained as volume/domain/uncertain, not reset after exclusions. Old C10k cannot be reused: requested source→volume→slice sampling differs from R5 source→patch. T0 reuse cannot be certified while annotation-mask semantics remain unresolved.\n')
 save(BASE/'reused_artifacts.json',dict(P1='initial teacher only; hash verified; no optimizer recovery',T0='not reused: unresolved mask pipeline',TJ='not reused: hierarchical sampling and identity audit differ',TM='not launched'))
 save(Path(os.environ['RUN_OUTPUT'])/'result.json',dict(status='complete',stage='S0 audit execution',ssl_gate='blocked',volumes=len(manifest),quarantined_train=sum(r['split']=='train' and not r['original_volume_id'] for r in manifest),P1_hash_match=True))
 print('S0 audit complete; SSL identity gate blocked',flush=True)
if __name__=='__main__':main()
