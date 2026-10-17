"""Run unchanged MV3 functions with new encoder registry and new output root."""
import argparse,gc,importlib.util,json,os,sys
from pathlib import Path
from common import R,M3,OLD,MV2,atomic,read,sha

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj

def setup():
 import torch
 out=R/'downstream';os.environ['MV3_ROOT']=str(out)
 data=module('data',M3/'src/data.py');data.OLD=OLD;data.MV2=MV2
 sys.path[:0]=[str(OLD/'src'),str(MV2/'src')]
 import mv_probe
 natural_build=mv_probe.build
 def build(model,seed):
  assert model in ['D1','D2'];net,spec=natural_build('P0',seed)
  checkpoint=R/'runs'/model/'formal/checkpoints/step000030000/teacher_backbone.pth'
  receipt=read(checkpoint.with_suffix('.json'));assert sha(checkpoint)==receipt['sha256']
  value=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);assert value['step']==30000
  assert value['official_weight_sha256']==spec['sha256'] and len(value['backbone'])==368
  net.dino_encoder.load_state_dict(value['backbone'],strict=True);net.dino_encoder.requires_grad_(False).eval()
  spec={**spec,'id':model,'checkpoint_path':str(checkpoint),'sha256':receipt['sha256']};del value;gc.collect();return net,spec
 mv_probe.build=build
 trainer=module('train',M3/'src/train.py')
 evaluator=module('d12_mv3_evaluate',M3/'src/evaluate.py')
 return trainer,evaluator,mv_probe

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','infer','score']);p.add_argument('--model',choices=['D1','D2']);p.add_argument('--path',type=Path);a=p.parse_args()
 trainer,evaluator,probe=setup();seed=20261014
 if a.mode=='train':
  out=R/'downstream/runs'/a.model/f'seed{seed}';out.mkdir(parents=True,exist_ok=True);trainer.train(a.model,seed,out)
 elif a.mode=='infer':evaluator.infer(a.model,seed)
 else:evaluator.score(a.path)
if __name__=='__main__':main()
