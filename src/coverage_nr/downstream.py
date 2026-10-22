"""Run unchanged MV3 functions with new encoder registry and new output root."""
import argparse,gc,importlib.util,json,os,sys
from pathlib import Path
from common import R,PARENT,M3,OLD,MV2,atomic,read,sha

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj

def setup():
 import torch
 # The separate100k output needs the same locked protocol, manifest and cache.
 if not (R/'evidence/readout_inputs_preflight.json').exists():
  from readout_inputs import prepare
  atomic(R/'evidence/readout_inputs_preflight.json',prepare(R,PARENT))
 out=R/'downstream';os.environ['MV3_ROOT']=str(out)
 data=module('data',M3/'src/data.py');data.OLD=OLD;data.MV2=MV2
 sys.path[:0]=[str(OLD/'src'),str(MV2/'src')]
 import mv_probe
 natural_build=mv_probe.build
 def build(model,seed):
  assert model in ['Coverage-NR'];net,spec=natural_build('P0',seed)
  checkpoint=R/'runs'/model/'formal/checkpoints/step000100000/teacher_backbone.pth'
  receipt=read(checkpoint.with_suffix('.json'));assert sha(checkpoint)==receipt['sha256']
  value=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);assert value['step']==100000
  assert value['official_weight_sha256']==spec['sha256'] and len(value['backbone'])==368
  net.dino_encoder.load_state_dict(value['backbone'],strict=True);net.dino_encoder.requires_grad_(False).eval()
  spec={**spec,'id':model,'checkpoint_path':str(checkpoint),'sha256':receipt['sha256']};del value;gc.collect();return net,spec
 mv_probe.build=build
 trainer=module('train',M3/'src/train.py')
 evaluator=module('d12_mv3_evaluate',M3/'src/evaluate.py')
 return trainer,evaluator,mv_probe

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','infer','score']);p.add_argument('--model',choices=['Coverage-NR']);p.add_argument('--path',type=Path);a=p.parse_args()
 if a.mode=='score':
  from gpu_scoring import accelerated
  with accelerated(a.path) as acceleration:
   _,evaluator,_=setup();evaluator.score(a.path)
   target=a.path.with_name(a.path.name.replace('_overlap.json','_metrics.json'))
   result=read(target);result.update(distance_backend='MV2_exact_GPU_EDT_MONAI',scoring_physical_gpu=acceleration['physical_gpu'],GPU_EDT_source_sha256=acceleration['source_sha256'],full_device_EDT_calls=acceleration['full_device_calls'],host_backed_EDT_calls=acceleration['host_backed_calls']);atomic(target,result)
  return
 trainer,evaluator,probe=setup();seed=20261014
 if a.mode=='train':
  out=R/'downstream/runs'/a.model/f'seed{seed}';out.mkdir(parents=True,exist_ok=True);trainer.train(a.model,seed,out)
 elif a.mode=='infer':evaluator.infer(a.model,seed)
if __name__=='__main__':main()
