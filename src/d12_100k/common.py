"""D1/D2 paths and immutable-input helpers; no historical writes."""
import hashlib,json,os
from pathlib import Path
R=Path(os.environ.get('D12_ROOT','/opt/data/3_117/tanghr/codex/ssl_diversity_d12_100k_20261018'))
BASE=R.parent.parent
M3=R.parent/'mv_generalist_v2_han_fullgt_20261014'
OLD=R.parent/'mv_generalist_stage1_20261013'
MV2=R.parent/'mv_dev_v2_notta_20261013'
B_FULL=R.parent/'mitoverse_full_stage1_20260928_163120'
PY=BASE/'envs/meddinov3/bin/python'
SSL_PY=BASE/'dinov3_shared/bin/python'
ZIP=BASE/'MitoNet/dataset/cem1.5m.zip'
WEIGHT=BASE/'DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
WEIGHT_SHA='8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def heartbeat(path,value):
 """Never expose a truncated heartbeat to concurrent training ranks."""
 path=Path(path);temporary=path.with_suffix('.heartbeat.tmp');temporary.write_text(str(value));temporary.replace(path)

def subtract_z(intervals,cut):
 out=[]
 for a,b in intervals:
  c,d=cut
  if d<=a or c>=b:out.append([a,b]);continue
  if a<c:out.append([a,c])
  if d<b:out.append([d,b])
 return out

def cem_exclusion(name):
 """Bounded identifiable-source cross-check, not exhaustive image deduplication."""
 low=name.lower()
 for term in ['muller2020','guay2020','wei2020_mitoem','kasthuri','lucchi','jrc_cos7','jrc_fly','jrc_mus-nacc']:
  if term in low:return 'known_or_related_MV3_dev_test_or_MB1_source:'+term
 return None

PARENT=Path('/opt/data/3_117/tanghr/codex/ssl_diversity_d12_20261017')
