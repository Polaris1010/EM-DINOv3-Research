"""Finalize the authorized experiment delivery after the last matched control succeeds."""
import subprocess,time,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent;PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
while not (BASE/'lower_lr_control_complete.json').exists():time.sleep(10)
for script in ['summarize.py','audit_optimizer.py','plot_results.py','build_report.py','package_results.py']:
 print('FINALIZE',script,flush=True);subprocess.run([PY,str(BASE/script)],cwd=BASE,check=True)
archive=BASE/'MitoTopo_CMCC_CMCT_Gram_validation.zip';h=hashlib.sha256(archive.read_bytes()).hexdigest()
(BASE/'delivery_ready.json').write_text(json.dumps({'status':'complete','archive':str(archive),'bytes':archive.stat().st_size,'sha256':h},indent=2));print('DELIVERY_READY',h,flush=True)
