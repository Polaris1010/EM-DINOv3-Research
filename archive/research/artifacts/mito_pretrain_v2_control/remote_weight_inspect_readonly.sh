set -euo pipefail
base=/opt/data/3_117/tanghr
"$base/envs/meddinov3/bin/python" - "$base/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" <<'PY'
import json,sys,torch
p=sys.argv[1]
x=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
print('top_level_type',type(x).__name__)
if isinstance(x,dict):
 print('top_level_keys',list(x)[:25])
 for key in ('model','teacher','student','state_dict'):
  if key in x and isinstance(x[key],dict):
   x=x[key];print('using_key',key);break
print('state_count',len(x))
for k in list(x)[:20]:
 v=x[k]; print('key',k,tuple(v.shape) if hasattr(v,'shape') else type(v).__name__)
for term in ('storage_tokens','cls_token','patch_embed.proj.weight','mask_token','attn.qkv.weight','rope'):
 matches=[(k,tuple(v.shape)) for k,v in x.items() if term in k and hasattr(v,'shape')]
 print('match',term,matches[:10])
PY
