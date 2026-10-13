import json,hashlib
from pathlib import Path
from urllib.request import urlopen,Request
out=Path(__file__).resolve().parent
commit='a2460c2547899283446e520b84523454f7e25608'
url=f'https://raw.githubusercontent.com/ricklisz/MedDINOv3/{commit}/nnUNet/nnunetv2/compute_metrics.py'
with urlopen(url,timeout=30) as f: data=f.read()
(out/'meddinov3_compute_metrics_upstream_20261012.py').write_bytes(data)
(out/'meddinov3_metrics_source_20261012.json').write_text(json.dumps({'commit':commit,'url':url,'sha256':hashlib.sha256(data).hexdigest()},indent=2))
print(commit,hashlib.sha256(data).hexdigest())
