import hashlib,json,shutil,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
def snapshot(out):
 target=Path(out)/'source_snapshot';target.mkdir(exist_ok=True);hashes={}
 for p in BASE.glob('*.py'):
  shutil.copy2(p,target/p.name);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 (Path(out)/'source_sha256.json').write_text(json.dumps(dict(files=hashes,python=sys.version),indent=2))
