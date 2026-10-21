"""Call the unchanged MV3 trainer/inference with four registered encoders and two new seeds."""
import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path
from common import R, M3, OLD, MV2, atomic, read, sha

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def setup():
    import torch
    torch.cuda.set_per_process_memory_fraction(0.40)
    os.environ['MV3_ROOT'] = str(R / 'downstream')
    data = module('data', M3 / 'src/data.py')
    data.OLD, data.MV2 = OLD, MV2
    sys.path[:0] = [str(OLD / 'src'), str(MV2 / 'src')]
    import mv_probe
    natural_build = mv_probe.build
    models = read(R / 'protocol.json')['models']

    def build(model, seed):
        spec = models[model]
        path = Path(spec['encoder_path'])
        assert sha(path) == spec['encoder_sha256']
        net, original = natural_build('P0', seed)
        value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert value['step'] == spec['ssl_steps']
        assert value['official_weight_sha256'] == original['sha256'] and len(value['backbone']) == 368
        net.dino_encoder.load_state_dict(value['backbone'], strict=True)
        net.dino_encoder.requires_grad_(False).eval()
        del value
        gc.collect()
        return net, {**original, 'id': model, 'checkpoint_path': str(path), 'sha256': spec['encoder_sha256']}

    mv_probe.build = build
    trainer = module('train', M3 / 'src/train.py')
    evaluator = module('seed_mv3_evaluate', M3 / 'src/evaluate.py')
    return trainer, evaluator, mv_probe

def smoke(model, seed):
    import torch
    trainer, _, probe = setup()
    # Original smoke's public names map to the requested registered encoder.
    build = trainer.build
    trainer.build = lambda ignored, s: build(model, s)
    out = R / 'smoke' / model / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    trainer.smoke('G-P0', seed, out)
    # The milestone monitor uses batches of16 native-grid tiles: test its actual peak too.
    net, _ = build(model, seed)
    import numpy as np
    probe.sliding(net, np.full((1024, 1024), 127, np.uint8), tile_batch_size=16)
    atomic(out / 'memory.json', {'status': 'PASS', 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'memory_fraction': 0.40})

def score(path):
    """Same exact physical evaluator on CPU; leaves VRAM to concurrent SSL/head jobs."""
    import math
    import time
    import numpy as np
    os.environ['MV3_ROOT'] = str(R / 'downstream')
    os.environ['MV2_GPU_EDT'] = '0'
    data = module('data', M3 / 'src/data.py')
    data.OLD, data.MV2 = OLD, MV2
    sys.path[:0] = [str(OLD / 'src'), str(MV2 / 'src')]
    import validation
    result = read(path)
    target = path.with_name(path.name.replace('_overlap.json', '_metrics.json'))
    if target.exists():
        assert read(target)['head_sha256'] == result['head_sha256']
        return
    row = next(r for r in data.manifest()['entries'] if r['id'] == result['region_id'])
    _, lab, valid, _ = data.open_arrays(row)
    gt = lab.astype(bool)
    valid = valid.astype(bool) if valid is not None else np.ones(gt.shape, bool)
    pred = np.load(path.with_name(path.name.replace('_overlap.json', '_prediction.npy')), mmap_mode='r') > 0
    assert pred.shape == gt.shape == valid.shape
    t0 = time.time()
    if not result['gt_foreground_voxels']:
        hd, ns = (0., 1.) if not result['prediction_foreground_voxels'] else (math.inf, 0.)
    elif not result['prediction_foreground_voxels']:
        hd, ns = math.inf, 0.
    else:
        hd, ns = validation.distances(pred, gt, valid, row['spacing_zyx_nm'])
    result.update(hd95_nm=hd, nsd_16nm=ns, metric_seconds=time.time()-t0,
                  distance_evaluator_source_sha256=sha(MV2 / 'src/validation.py'),
                  distance_backend='MV2_existing_exact_CPU_EDT_MONAI')
    atomic(target, result)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['smoke', 'train', 'infer', 'score'])
    p.add_argument('--model')
    p.add_argument('--seed', type=int)
    p.add_argument('--path', type=Path)
    a = p.parse_args()
    if a.mode == 'score':
        score(a.path)
    elif a.mode == 'smoke':
        smoke(a.model, a.seed)
    else:
        assert a.seed in read(R / 'protocol.json')['new_seeds']
        trainer, evaluator, _ = setup()
        if a.mode == 'infer':
            evaluator.infer(a.model, a.seed)
        else:
            out = R / 'downstream/runs' / a.model / f'seed{a.seed}'
            out.mkdir(parents=True, exist_ok=True)
            trainer.train(a.model, a.seed, out)
