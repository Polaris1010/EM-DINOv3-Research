set -euo pipefail
root=/opt/data/3_117/tanghr/MitoVerse_data
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
catalog = json.loads((root/'catalog.json').read_text())
for vid in [sorted(catalog)[0], next(x for x in sorted(catalog) if x.startswith('openorganelle'))]:
    row = catalog[vid]
    volume = root/row['zarr']
    print(json.dumps({'volume_id':vid,'catalog_row':row,'group_attrs':json.loads((volume/'.zattrs').read_text()),'image_meta':json.loads((volume/'img'/'.zarray').read_text())},default=str))
PY
sha256sum /opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
git -C /opt/data/3_117/tanghr/DINOv3/dinov3 rev-parse HEAD
