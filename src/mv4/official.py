"""Run unchanged official Engine2d/Preprocessor definitions without Qt imports.

The AST-selected definitions are compiled verbatim from hash-locked upstream
files. Only imports and the local model path are supplied by this headless adapter.
No inference algorithm, normalization, threshold or postprocessing is rewritten.
"""
import ast,os
from common import R,read,sha,config

def definitions(path,names,namespace):
 tree=ast.parse(path.read_text())
 chosen=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 assert {n.name for n in chosen}==set(names)
 exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec'),namespace)

def build():
 import numpy as np,torch,yaml
 from tqdm import tqdm
 from empanada.data.utils import resize_by_factor
 from empanada.inference.engines import PanopticDeepLabRenderEngine
 from empanada.inference import rle
 from empanada.inference.tile import Tiler
 from empanada.consensus import merge_objects_from_tiles,merge_semantic_from_tiles
 c=config();locks=read(R/'assets/official_assets.json')
 for name in ['inference_upstream.py','utils_upstream.py','model_config.yaml','MitoNet_v1.pth']:
  assert sha(R/'assets'/name)==locks['files'][name]['sha256'],name
 ns=dict(np=np,torch=torch,os=os,tqdm=tqdm,resize_by_factor=resize_by_factor,
  PanopticDeepLabRenderEngine=PanopticDeepLabRenderEngine,rle=rle,Tiler=Tiler,
  merge_objects_from_tiles=merge_objects_from_tiles,merge_semantic_from_tiles=merge_semantic_from_tiles)
 definitions(R/'assets/utils_upstream.py',{'normalize','to_tensor','Preprocessor','load_model_to_device'},ns)
 definitions(R/'assets/inference_upstream.py',{'Engine2d'},ns)
 model_config=yaml.safe_load((R/'assets/model_config.yaml').read_text())
 model_config['model']=str(R/'assets/MitoNet_v1.pth')
 assert model_config['norms']=={'mean':0.57571,'std':0.12765} and model_config['labels']==[1]
 engine=ns['Engine2d'](model_config,**c['engine_parameters'])
 engine.engine.model.requires_grad_(False).eval()
 return engine
