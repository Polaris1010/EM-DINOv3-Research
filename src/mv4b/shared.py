"""Read-only reuse of the locked MV4 inputs and official environment."""
import os,sys
from pathlib import Path
M4=Path('/opt/data/3_117/tanghr/codex/mv4_official_mitonet_20261014')
os.environ['MV4_ROOT']=str(M4)
sys.path.insert(0,str(M4/'src'))
import common as legacy
from common import M3,MV2,read,sha,atomic,rows,cache,image_array,evaluation_arrays,gpu_guard
R=Path(os.environ.get('MV4B_ROOT','/opt/data/3_117/tanghr/codex/mv4b_semantic_panoptic_20261014'))
MODEL='Official-MitoNet-Semantic-v1'
def config():
 c=read(R/'MV4B_protocol.json')
 assert sha(M4/'MV4_protocol.json')==c['mv4_protocol_sha256']
 assert c['confidence_thr']==legacy.config()['engine_parameters']['confidence_thr']==0.5
 assert c['normalization']=={'mean':0.57571,'std':0.12765}
 assert c['manifest_sha256']==legacy.config()['manifest_sha256']
 return c
def output():return R/'runs'/MODEL
