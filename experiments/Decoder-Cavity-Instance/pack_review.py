from pathlib import Path
import gzip,tarfile,json,hashlib
p=Path(__file__).resolve().parent
files=[*p.glob('*.py'),p/'PROTOCOL.md',p/'audit.json']+[p/n for n in ('EXPERIMENT_SUMMARY.md','semantic_and_cavity.csv','instance_metrics.csv','hq_per_image.csv','hq_summary.csv','core_mb1_comparison.csv')]
for a in sorted(p.glob('arm_*')):
    if not a.is_dir():continue
    files += [a/n for n in ('trained.json','evaluation_pass.json','smoke_pass.json','train_config.json')]
    files += [p/f'{a.name}.{label}.log' for label in ('smoke','train','eval')]
    if (a/'watershed_locked.json').exists():files.append(a/'watershed_locked.json')
assert len(list((p/'visual_panels').glob('*.png')))==12
assert len(list((p/'visual_panels_extra').glob('*.png')))==2
blob=p/'per_plane_scores.jsonl.gz'
n=0
with gzip.open(blob,'wb') as h:
    for f in sorted(p.glob('arm_*/evaluations/*/*/*.jsonl')):
        content=f.read_bytes()
        h.write(content)
        n+=content.count(b'\n')
print('metric records',n,flush=True)
files.append(blob)
target=p/'argon_decoder_final.tar.gz'
tmp=target.with_suffix('.tmp')
with tarfile.open(tmp,'w:gz') as h:
    for f in files:h.add(f,arcname=str(f.relative_to(p)),recursive=False)
tmp.replace(target)
print(target.stat().st_size,hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
