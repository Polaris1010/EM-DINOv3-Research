set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" <<'PY'
import json,sys,torch
from pathlib import Path
r=Path(sys.argv[1]);sys.path.insert(0,str(r/'src'))
from seg_model import load_encoder,SegModel
torch.set_num_threads(4)
root=Path('/opt/data/3_117/tanghr')
weight=root/'DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
encoder=load_encoder(root/'DINOv3/dinov3',weight)
model=SegModel(encoder,0,False)
with torch.inference_mode():
    features=model.encoder.get_intermediate_layers(torch.zeros(1,3,32,32),n=[5,11,17,23],reshape=True)
result={'status':'complete','storage_tokens':model.encoder.n_storage_tokens,
        'feature_shapes':[list(v.shape) for v in features],
        'decoder_parameters':sum(p.numel() for p in model.decoder.parameters()),
        'frozen_encoder':all(not p.requires_grad for p in model.encoder.parameters())}
assert result['feature_shapes']==[[1,1024,2,2]]*4 and result['frozen_encoder']
(r/'evidence/seg_encoder_cpu_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
