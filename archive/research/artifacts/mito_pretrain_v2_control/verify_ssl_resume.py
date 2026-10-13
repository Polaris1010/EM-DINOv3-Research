"""Compare a split 5+5 run against an uninterrupted 10-step run."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=2e-5)
    args = parser.parse_args()
    a = torch.load(args.continuous, map_location='cpu', mmap=True, weights_only=False)
    b = torch.load(args.resumed, map_location='cpu', mmap=True, weights_only=False)
    result = {'status': 'complete', 'continuous': str(args.continuous), 'resumed': str(args.resumed),
              'step_equal': a['completed_steps'] == b['completed_steps'] == 10,
              'sample_chain_equal': a['consumption_chain'] == b['consumption_chain'],
              'cfg_equal': a['cfg'] == b['cfg'], 'manifest_equal': a['manifest_sha'] == b['manifest_sha'],
              'max_abs_diff': {}, 'unequal_tensors': {}, 'relative_l2_diff': {},
              'absolute_tolerance': args.tolerance}
    for part in ('student', 'teacher', 'dino_loss', 'ibot_loss'):
        assert a[part].keys() == b[part].keys(), part
        unequal = 0
        largest = 0.0
        diff_sq = 0.0
        reference_sq = 0.0
        for name in a[part]:
            x, y = a[part][name], b[part][name]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise AssertionError(f'{part}.{name} shape/dtype mismatch')
            if not torch.equal(x, y):
                unequal += 1
                if x.is_floating_point():
                    difference = (x.float()-y.float()).double()
                    largest = max(largest, float(difference.abs().max()))
                    diff_sq += float(difference.square().sum())
                    reference_sq += float(x.double().square().sum())
        result['unequal_tensors'][part] = unequal
        result['max_abs_diff'][part] = largest
        result['relative_l2_diff'][part] = (diff_sq/reference_sq)**0.5 if reference_sq else 0.0
    result['bitwise_equal'] = all(v == 0 for v in result['unequal_tensors'].values())
    result['numerically_consistent'] = all(v <= args.tolerance for v in result['max_abs_diff'].values())
    print(json.dumps(result))
    if not all(result[k] for k in ('step_equal', 'sample_chain_equal', 'cfg_equal', 'manifest_equal')):
        return 2
    if not result['numerically_consistent']:
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
