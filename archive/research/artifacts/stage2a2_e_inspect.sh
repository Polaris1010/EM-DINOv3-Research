set -e
BASE=/opt/data/3_117/tanghr
CK="$BASE/codex/mitoverse_full_stage1_20260928_163120/runs/B_full/formal_attempt1/checkpoints/step000100000/teacher_backbone.pth"
printf 'metadata\n'
cat "$BASE/codex/mitoverse_full_stage1_20260928_163120/runs/B_full/formal_attempt1/checkpoints/step000100000/metadata.json"
printf '\nsplits\n'
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('/opt/data/3_117/tanghr/Med_DINOv3/nnUNet_preprocessed')
for ds in ('Dataset111_CMCC','Dataset131_CMCT'):
    p=root/ds/'splits_final.json'
    folds=json.loads(p.read_text())
    f=folds[0]
    print(ds,'folds',len(folds),'train',len(f['train']),'val',len(f['val']),'disjoint',not set(f['train'])&set(f['val']))
    print('first_cases',f['train'][:2],f['val'][:2])
PY
printf '\ncheckpoint_keys\n'
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$CK" <<'PY'
import sys,torch
p=sys.argv[1]
x=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
print('top_type',type(x).__name__)
print('top_keys',list(x)[:12] if hasattr(x,'keys') else None)
sd=x.get('backbone',x.get('model',x.get('teacher',x))) if isinstance(x,dict) else x
print('state_keys',len(sd),'first',[(k,tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in list(sd.items())[:12]])
for k in ('patch_embed.proj.weight','pos_embed','storage_tokens','register_tokens','blocks.0.attn.qkv.weight','blocks.0.attn.qkv.bias','norm.weight'):
    if k in sd: print('shape',k,tuple(sd[k].shape))
PY
