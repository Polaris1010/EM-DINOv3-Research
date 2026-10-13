import getpass,json
from pathlib import Path
import paramiko

base=Path(__file__).resolve().parent
out=base/'p1_mitonet_analysis_20260927'
out.mkdir(exist_ok=True)
client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.203.97',username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=60)
try:
 script=(base/'audit_p1_mitonet_remote.py').read_text()
 cmd="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY_AUDIT'\n"+script+"\nPY_AUDIT\n"
 _,stdout,stderr=client.exec_command(cmd)
 raw=stdout.read();error=stderr.read().decode();code=stdout.channel.recv_exit_status()
 print(error)
 if code:raise RuntimeError(f'Remote audit failed: {code}')
 data=json.loads(raw);(out/'audited_evidence.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
 print(f'Saved {len(data["rows"])} volume-model rows; {len(data["center_slice_checks"])} center slice checks to {out}')
finally:client.close()
