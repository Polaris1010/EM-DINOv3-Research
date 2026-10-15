"""All seven datasets, no GT access; original MV3 sliding or MV4 Engine2d."""
import argparse,gc,os,sys,time
from pathlib import Path
import numpy as np
from common import R,OLD,M3,M4,read,sha,atomic,config,rows,run_spec,gpu_guard,Planes
def dinov3(spec):
 import torch
 sys.path.insert(0,str(OLD/'src'))
 from mv_probe import build,sliding,state_hash,head_state
 net,enc=build(spec['encoder_id'],spec['seed']);assert enc['sha256']==spec['encoder_sha256']
 allowed=[np.core.multiarray.scalar,np.dtype,type(np.dtype(np.float64))]
 with torch.serialization.safe_globals(allowed):saved=torch.load(spec['checkpoint_path'],map_location='cpu',weights_only=True,mmap=True)
 assert saved['step']==40000 and saved['metadata']['model']==spec['model'] and saved['metadata']['seed']==spec['seed']
 missing,extra=net.load_state_dict(saved['head'],strict=False);assert not extra and all(k.startswith('dino_encoder.') for k in missing)
 assert state_hash(head_state(net))==state_hash(saved['head']);net.requires_grad_(False).eval();del saved;gc.collect()
 return lambda plane:sliding(net,plane,tile_batch_size=16)
def mitonet(spec):
 import ast,torch,yaml
 from tqdm import tqdm
 from empanada.data.utils import resize_by_factor
 from empanada.inference.engines import PanopticDeepLabRenderEngine
 from empanada.inference import rle
 from empanada.inference.tile import Tiler
 from empanada.consensus import merge_objects_from_tiles,merge_semantic_from_tiles
 ns=dict(np=np,torch=torch,os=os,tqdm=tqdm,resize_by_factor=resize_by_factor,PanopticDeepLabRenderEngine=PanopticDeepLabRenderEngine,rle=rle,Tiler=Tiler,merge_objects_from_tiles=merge_objects_from_tiles,merge_semantic_from_tiles=merge_semantic_from_tiles)
 locks=read(M4/'assets/official_assets.json')
 for fname,names in [('utils_upstream.py',{'normalize','to_tensor','Preprocessor','load_model_to_device'}),('inference_upstream.py',{'Engine2d'})]:
  p=M4/'assets'/fname;assert sha(p)==locks['files'][fname]['sha256'];tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert {n.name for n in nodes}==names;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 assert sha(M4/'assets/model_config.yaml')==locks['files']['model_config.yaml']['sha256']
 mc=yaml.safe_load((M4/'assets/model_config.yaml').read_text());mc['model']=spec['checkpoint_path']
 engine=ns['Engine2d'](mc,**config()['mitonet_engine_parameters'])
 for parameter in engine.engine.model.parameters():parameter.requires_grad_(False)
 engine.engine.model.eval();return engine.infer
def main(run_id,smoke=False):
 gpu=gpu_guard();spec=run_spec(run_id);locks=read(R/'evidence/checkpoint_verification.json')['assets']
 for p in [spec['checkpoint_path']]+([spec['encoder_path']] if spec['family']=='dinov3' else []):
  st=Path(p).stat();assert [st.st_size,st.st_mtime_ns]==[locks[p]['bytes'],locks[p]['mtime_ns']]
 infer=dinov3(spec) if spec['family']=='dinov3' else mitonet(spec);out=R/('smoke/'+run_id if smoke else 'runs/'+run_id);out.mkdir(parents=True,exist_ok=True);started=time.time()
 selected=rows()
 if smoke:selected=[selected[0],selected[4],selected[5],selected[6]]
 for row in selected:
  target=out/(row['id']+'_inference.json')
  if target.exists():
   old=read(target);assert old['protocol_sha256']==sha(R/'MB1_protocol.json') and old['checkpoint_sha256']==spec['checkpoint_sha256'];continue
  image=Planes(row['image_path'],row['shape_zyx']);shape=[1,*row['shape_zyx'][1:]] if smoke else row['shape_zyx'];n=shape[0]
  pred=np.lib.format.open_memmap(out/(row['id']+'_prediction.npy'),mode='w+',shape=tuple(shape),dtype=np.uint8)
  pan=np.lib.format.open_memmap(out/(row['id']+'_panoptic.npy'),mode='w+',shape=tuple(shape),dtype=np.uint32) if spec['family']=='mitonet' else None
  for z in range(n):
   original=np.asarray(image[z]);assert original.dtype==np.uint8 and original.shape==tuple(shape[1:]);seg=infer(original);assert seg.shape==original.shape
   if pan is not None:assert seg.min()>=0;pan[z]=seg.astype(np.uint32)
   pred[z]=seg>0
   if z%25==0:atomic(out/'progress.json',{'run_id':run_id,'model':spec['model'],'seed':spec['seed'],'dataset':row['dataset'],'image':row['id'],'plane':z+1,'planes':n,'seconds':time.time()-started,'physical_gpu':gpu});print('MB1_INFER',run_id,row['id'],z+1,n,flush=True)
  pred.flush()
  if pan is not None:pan.flush()
  del pred;image.close()
  atomic(target,{'status':'complete','run_id':run_id,'model':spec['model'],'seed':spec['seed'],'dataset_id':row['dataset_id'],'dataset':row['dataset'],'image_id':row['id'],'shape_zyx':shape,'all_native_planes':not smoke,'checkpoint_sha256':spec['checkpoint_sha256'],'protocol_sha256':sha(R/'MB1_protocol.json'),'manifest_sha256':sha(R/'MB1_benchmark_manifest.json'),'gt_read_by_inference':False,'tta':False,'physical_gpu':gpu,'panoptic_retained':spec['family']=='mitonet','instance_interpretation':'original per-plane panoptic IDs; no3D identity or instance metric claim' if pan is not None else 'semantic only'})
 atomic(out/'inference_complete.json',{'status':'PASS' if smoke else 'complete','run_id':run_id,'entries':len(selected),'datasets':7 if not smoke else 4,'planes':2595 if not smoke else 4,'physical_gpu':gpu,'checkpoint_sha256':spec['checkpoint_sha256'],'protocol_sha256':sha(R/'MB1_protocol.json'),'seconds':time.time()-started})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.run_id,a.smoke)
