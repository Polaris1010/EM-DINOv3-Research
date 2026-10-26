"""Build the credential-free, new-directory-only deployment script."""
import base64,json
from pathlib import Path

def main():
 r=Path(__file__).resolve().parents[1]
 scratch=r.parent/'artifacts/coverage_nr_20261021'
 payload={'code/'+p.name:base64.b64encode(p.read_bytes()).decode() for p in (r/'src/coverage_nr').glob('*.py')}
 for local,dest in [('configs/pretrain/D12_100k_base.yaml','D12_base.yaml'),('configs/pretrain/Coverage_NR_protocol.json','D12_protocol.json')]:
  payload[dest]=base64.b64encode((r/local).read_bytes()).decode()
 script="""set -eu
python3 - <<'PY'
import base64,pathlib,json,shutil,os
r=pathlib.Path('/opt/data/3_117/tanghr/codex/coverage_nr_p0_20261021');old=r.parent/'ssl_diversity_d12_20261017'
assert not r.exists();r.mkdir()
for name in ['code','plan','logs','evidence','cache','downstream','benchmark']:(r/name).mkdir()
payload=PAYLOAD
for name,data in payload.items():(r/name).write_bytes(base64.b64decode(data))
for name in ['D1_manifest.json','mitoverse_pool.json','cem_pool.json']:shutil.copyfile(old/name,r/name)
d=json.loads((old/'D2_manifest.json').read_text());d['coverage_plan_sha256']='PLAN_SHA';d['experiment_id']='Coverage-NR-P0-100k';(r/'D2_manifest.json').write_text(json.dumps(d,indent=2)+'\\n')
os.symlink(old/'inputs',r/'inputs',target_is_directory=True)
shutil.copyfile(old/'code/dinov3_commit.txt',r/'code/dinov3_commit.txt')
(r/'CODE_COMMIT').write_text('PREPARING\\n')
for f in (r/'code').glob('*.py'):compile(f.read_text(),str(f),'exec')
(r/'STATUS.json').write_text(json.dumps({'status':'prepared','training_started':False})+'\\n')
print(json.dumps({'root':str(r),'prepared':True,'source_files':len(payload)}))
PY
""".replace('PAYLOAD',repr(payload)).replace('PLAN_SHA',json.loads((scratch/'plan/plan.json').read_text())['id_sha256'])
 (scratch/'deploy.sh').write_text(script,encoding='utf-8',newline='\n')
if __name__=='__main__':main()
