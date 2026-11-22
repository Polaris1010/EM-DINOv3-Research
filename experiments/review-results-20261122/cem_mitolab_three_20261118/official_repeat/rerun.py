from pathlib import Path
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean

O = Path(__file__).resolve().parent
X = O.parent
MB = X.parent / 'mb1_official_benchmark_20261015'
sys.path.insert(0, str(MB / 'code'))
from common import Planes, atomic, config, gpu_guard, rows, sha

SEED = 20261014
MODELS = ['P0', 'B100', 'Barrier100']
METRICS = ['dice', 'iou', 'precision', 'recall']


def evaluate(rank):
    import numpy as np
    import tifffile
    import torch
    from infer import mitonet

    assert gpu_guard() == rank + 1
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    spec, = [r for r in config()['runs'] if r['model'] == 'Official-MitoNet-v1']
    weight = sha(spec['checkpoint_path'])
    assert weight == spec['checkpoint_sha256']
    assert sha(X / 'manifest.json') == 'b318d48f3716577645232c1ecaad846fba229101f4f1ea411b9f2edb198d46b3'
    infer = mitonet(spec)
    scores = []

    @torch.inference_mode()
    def measure(image, gt, split, group, name, z):
        assert image.dtype == np.uint8 and image.shape == gt.shape and gt.min() >= 0
        pan = np.asarray(infer(image))
        assert pan.shape == image.shape and pan.min() >= 0
        pred = torch.as_tensor(pan > 0, device='cuda')
        truth = torch.as_tensor(gt > 0, device='cuda')
        tp, pos, detected = torch.stack([(pred & truth).sum(), truth.sum(), pred.sum()]).tolist()
        fp, fn = detected - tp, pos - tp
        union = tp + fp + fn
        scores.append(dict(split=split, group=group, name=name, z=z, tp=tp, fp=fp, fn=fn,
                           tn=image.size-union, dice=2*tp/(pos+detected) if pos+detected else 1.,
                           iou=tp/union if union else 1., precision=tp/detected if detected else None,
                           recall=tp/pos if pos else None))
        if len(scores) % 100 == 0:
            print(split, group, len(scores), flush=True)

    manifest = json.loads((X / 'manifest.json').read_text())
    data = X.parent.parent / 'MitoNet/dataset/EMPIAR-11037/cem_mitolab'
    index = 0
    for g in manifest['validation']:
        for name, shape, foreground in manifest['groups'][g]:
            if index % 3 == rank:
                a = tifffile.imread(data / g / 'images' / name)
                m = tifffile.imread(data / g / 'masks' / name)
                assert list(a.shape) == shape and np.count_nonzero(m) == foreground
                measure(a, m, 'cem_validation', g, name, 0)
            index += 1
    for row in rows():
        a = Planes(row['image_path'], row['shape_zyx'])
        m = Planes(row['label_path'], row['shape_zyx'])
        for z in range(rank, row['shape_zyx'][0], 3):
            measure(a[z], m[z], 'mb1', row['dataset'], row['id'], z)
        a.close()
        m.close()
    atomic(O / f'shard{rank}.json', dict(weight=weight, seed=SEED, gpu=rank+1, scores=scores))


def summarize():
    shards = [json.loads((O / f'shard{i}.json').read_text()) for i in range(3)]
    assert len({s['weight'] for s in shards}) == 1
    fresh = [r for s in shards for r in s['scores']]
    def key(r):
        return r['group'], r['name'], r.get('z', 0)
    stats = []
    for split, n in [('cem_validation', 1424), ('mb1', 2595)]:
        measured = [r for r in fresh if r['split'] == split]
        keys = {key(r) for r in measured}
        assert len(keys) == len(measured) == n
        truth = {key(r): (r['tp']+r['fn'],r['tn']+r['fp']) for r in measured}
        for model in MODELS + ['Official-MitoNet-v1']:
            records = measured if model == 'Official-MitoNet-v1' else json.loads((X / model / (split+'.json')).read_text())
            assert len(records) == n and {key(r): (r['tp']+r['fn'],r['tn']+r['fp']) for r in records} == truth
            groups = defaultdict(list)
            for r in records: groups[r['group']].append(r)
            for g, rs in sorted(groups.items()):
                line = dict(split=split, model=model, group=g, images=len(rs))
                for k in METRICS:
                    vals = [r[k] for r in rs if r[k] is not None]
                    line[k] = mean(vals) if vals else None
                stats.append(line)
    with (O / 'comparison.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=stats[0])
        w.writeheader()
        w.writerows(stats)
    historical = list(csv.DictReader((MB / 'MB1_per_dataset_model_means.csv').open()))
    def values(split,group):
        return [mean(r['dice'] for r in stats if r['split']==split and r['model']==model and
                     (group is None or r['group']==group)) for model in MODELS+['Official-MitoNet-v1']]
    lines = ['# Official MitoNet fresh inference', '', 'Seed 20261014, official Engine2d, scale 1, confidence 0.5, TTA off. Frozen released checkpoint SHA256 '+shards[0]['weight']+'.',
             '', '| Dataset | P0 | B100 | Barrier100 | Official MitoNet rerun | Barrier minus MitoNet |','|---|---:|---:|---:|---:|---:|']
    for split,group in [('cem_validation',None),('mb1',None)]+[('mb1',g) for g in sorted({r['group'] for r in stats if r['split']=='mb1'})]:
        v=values(split,group)
        lines.append('| '+(group or split)+' | '+' | '.join(f'{x:.6f}' for x in [*v,v[2]-v[3]])+' |')
    diff = [abs(float(r['dice'])-values('mb1',r['dataset'])[3]) for r in historical if r['model']=='Official-MitoNet-v1']
    assert len(diff)==7
    lines+=['',f'Maximum absolute group Dice difference from archived rounded MB-1 metrics: {max(diff):.9g}.',
            'Identical image IDs, per-image foreground and background counts verified across all four systems. CEM holdout split was defined for the three new heads. Original MitoNet supervised exposure to these images is possible. The inference seed does not alter its historical training seed.']
    (O/'rerun_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='report': summarize()
    else: evaluate(int(sys.argv[1]))
