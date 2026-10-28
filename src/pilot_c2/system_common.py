"""New system output roots, with the accepted historical data/readout assets."""
import os,json,hashlib,sys,importlib.util
from pathlib import Path
ROOT=Path(os.environ['PILOT_C2_ROOT']);B=ROOT.parent
MODEL=os.environ['PILOT_C2_MODEL'];assert MODEL in ['C0_Vanilla_45k','C1_Barrier_45k']
R=ROOT/'systems'/MODEL
PA=B/'feature_mechanism_pilot_a_20261027';A2=B/'pilot_a2_readout_calibration_20261027'
PARENT=B/'ssl_diversity_d12_20261017';LATE=B/'ssl_diversity_d12_100k_20261018'
M3=B/'mv_generalist_v2_han_fullgt_20261014';OLD=B/'mv_generalist_stage1_20261013';MV2=B/'mv_dev_v2_notta_20261013'
PY=B.parent/'envs/meddinov3/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def model_spec(model):
 assert model==MODEL
 return read(R/'encoder.json')
