set -euo pipefail
pkill -f '^find /root/.cache/huggingface -maxdepth 3 -type f' || true
for dir in /root/.cache /root/.cache/huggingface /root/.cache/huggingface/hub /root/.cache/torch/hub/checkpoints /opt/data/3_117/tanghr/microsam/micro-sam; do
 if test -d "$dir"; then echo "DIR $dir"; ls -lah "$dir" | head -30; fi
done
/opt/data/3_117/tanghr/envs/microsam/bin/python - <<'PY'
import micro_sam,inspect,micro_sam.util
print('micro_sam',micro_sam.__file__)
print('get_sam_model',inspect.signature(micro_sam.util.get_sam_model))
PY
