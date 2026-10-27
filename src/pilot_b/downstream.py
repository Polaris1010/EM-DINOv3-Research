"""Unchanged official MV3 training/evaluation with one new encoder/output registry."""
import argparse,gc,os,sys,shutil
from pathlib import Path
from common import R,PARENT,M3,OLD,MV2,MODEL,read,atomic,sha,module,model_spec
def prepare():
 out=R/'downstream';out.mkdir(exist_ok=True);source=PARENT/'downstream'
 for name in ['MV-Generalist.json','src/data.py']:
  p=out/name;p.parent.mkdir(exist_ok=True)
  if p.exists():assert sha(p)==sha(source/name)
  else:shutil.copyfile(source/name,p)
 original=read(source/'MV3_protocol.json');protocol={**original,'models':{MODEL:MODEL}}
 if (out/'MV3_protocol.json').exists():assert read(out/'MV3_protocol.json')==protocol
 else:atomic(out/'MV3_protocol.json',protocol)
 for k in original:
  if k!='models':assert original[k]==protocol[k]
 cache=(source/'cache').resolve();link=out/'cache'
 if not link.exists():os.symlink(cache,link,target_is_directory=True)
 assert link.resolve()==cache
 rows=read(out/'MV-Generalist.json')['entries'];assert sum(x['supervised_role']=='train' for x in rows)==156 and sum(x['supervised_role']=='dev' for x in rows)==38
 receipt={'manifest_sha256':sha(out/'MV-Generalist.json'),'parent_protocol_sha256':sha(source/'MV3_protocol.json'),'new_protocol_sha256':sha(out/'MV3_protocol.json'),'changed_fields':['models'],'cache':str(cache),'train':156,'dev':38,'historical_inputs_read_only':True}
 if (R/'evidence/readout_inputs.json').exists():assert read(R/'evidence/readout_inputs.json')==receipt
 else:atomic(R/'evidence/readout_inputs.json',receipt)
def setup():
 import torch
 prepare();os.environ['MV3_ROOT']=str(R/'downstream');data=module('data',M3/'src/data.py');data.OLD=OLD;data.MV2=MV2
 sys.path[:0]=[str(OLD/'src'),str(MV2/'src')]
 import mv_probe
 natural=mv_probe.build
 def build(model,seed):
  assert model in [MODEL,'D2-30k','D2-100k'];net,spec=natural('P0',seed);s=model_spec(model)
  assert sha(s['path'])==s['sha256'];v=torch.load(s['path'],map_location='cpu',weights_only=True,mmap=True)
  assert v['official_weight_sha256']==spec['sha256'] and len(v['backbone'])==368
  net.dino_encoder.load_state_dict(v['backbone'],strict=True);net.dino_encoder.requires_grad_(False).eval()
  assert mv_probe.state_hash(mv_probe.head_state(net))==read(R/'protocol.json')['decoder_initial_sha256']
  result={**spec,'id':model,'checkpoint_path':s['path'],'sha256':s['sha256']};del v;gc.collect();return net,result
 mv_probe.build=build;trainer=module('train',M3/'src/train.py');evaluator=module('pilot_b_mv3_evaluate',M3/'src/evaluate.py')
 return trainer,evaluator,mv_probe
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','infer','score']);p.add_argument('--path',type=Path);a=p.parse_args()
 if a.mode=='score':
  from gpu_scoring import accelerated
  with accelerated(a.path):
   _,e,_=setup();e.score(a.path)
  return
 t,e,_=setup();out=R/'downstream/runs'/MODEL/'seed20261014';out.mkdir(parents=True,exist_ok=True)
 if a.mode=='train':t.train(MODEL,20261014,out)
 else:e.infer(MODEL,20261014)
if __name__=='__main__':main()
