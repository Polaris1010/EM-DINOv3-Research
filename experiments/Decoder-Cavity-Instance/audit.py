import csv,json,math,statistics as s
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parent
armed=('arm_a_multiscale_sem','arm_b_multiscale_bcd')
settings=('stride120','baseline128')
ref={}
counts={}
for a in armed:
 for w in settings:
    top=root/a/'evaluations'/w
    ids=defaultdict(set)
    vals=defaultdict(list)
    for f in top.glob('*/*.jsonl'):
        for line in f.read_text().splitlines():
            r=json.loads(line);key=(r['id'],r['z']);split=r['split']
            assert key not in ids[split]
            ids[split].add(key)
            if split=='mb1':
                assert all(q['closed_hole_px']+q['open_cavity_px']==q['interior_fn_px'] for q in r['cavity'])
                vals[r['group']].append(r)
    assert len(list((top/'mv3').glob('*.jsonl')))==38
    assert [len(ids[k]) for k in ('mb1','cem_validation','hq')]==[2595,1424,20]
    if not ref:ref=ids
    assert ref==ids
    dice=s.mean(s.mean(r['semantic']['dice'] for r in entries) for entries in vals.values())
    ratio=s.mean(sum(r['cavity'][0]['interior_fn_px'] for r in v)/sum(r['cavity'][0]['interior_px'] for r in v) for v in vals.values())
    with (root/'semantic_and_cavity.csv').open() as f:
        record=next(r for r in csv.DictReader(f) if r['arm']==a and r['window']==w and r['split']=='mb1' and r['group']=='ALL')
    assert math.isclose(float(record['dice']),dice,abs_tol=1e-13)
    assert math.isclose(float(record['interior_fn_ratio_r3']),ratio,abs_tol=1e-13)
    counts[f'{a}/{w}']={k:len(v) for k,v in ids.items()}
assert len(list((root/'visual_panels').glob('*.png')))==12
assert len(list((root/'visual_panels_extra').glob('*.png')))==2
assert len(list((root/'hq_png').rglob('*.png')))==80
(root/'audit.json').write_text(json.dumps(dict(status='PASS',coverage=counts,matching_unit_ids=True,independent_MB1_Dice_and_r3=True,paired_panels=12,extra_fly_brain=2,hq_png=80),indent=2))
