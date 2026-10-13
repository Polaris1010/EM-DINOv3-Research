set -euo pipefail
for dir in /root/.cache/torch/hub/checkpoints /root/.cache/micro_sam /root/.cache/microsam /root/.cache/huggingface /opt/data/3_117/tanghr/microsam/micro-sam/checkpoints /opt/data/3_117/tanghr/microsam/micro-sam/models /opt/data/3_117/tanghr/MedSAM/checkpoints /opt/data/3_117/tanghr/MedSAM/weights; do
 if test -d "$dir"; then
  echo "DIR $dir"
  find "$dir" -maxdepth 3 -type f \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.safetensors' -o -iname '*.ckpt' \) -printf '%p %s bytes\n' | head -30
 fi
done
/opt/data/3_117/tanghr/envs/microsam/bin/python - <<'PY'
import micro_sam,inspect,micro_sam.util
print('micro_sam',micro_sam.__file__)
print('util',micro_sam.util.__file__)
print('get_sam_model',inspect.signature(micro_sam.util.get_sam_model))
PY
rg -n 'sam_vit_[hbl]|vit_b|vit_l|vit_h|checkpoint' /opt/data/3_117/tanghr/microsam/micro-sam/micro_sam/util.py | head -45 || true
