"""Guarded batch assignment for the remaining three 100-step SSL gates."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import time

from guard import Guard, save
from settings import BASE

ASSIGNMENTS = [('L', 1), ('T', 2), ('LT', 3)]


def assess(path):
    result = json.loads((path/'result.json').read_text())
    steps = [json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
    assert result['status'] == 'complete' and result['completed_steps'] == 100
    assert len(steps) == 100 and [r['step'] for r in steps] == list(range(1, 101))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['grad_norm'] > 0 for r in steps)
    assert all(r['feature_stat']['cls_batch_std_mean'] > 1e-4 and
               r['feature_stat']['patch_spatial_std_mean'] > 1e-4 for r in steps)
    assert any(v > 0 for v in steps[-1]['tracked_parameter_delta_l2'].values())
    return {'arm': result['arm'], 'gpu': result.get('gpu'), 'completed_steps': 100,
            'sample_chain': result['sample_chain'],
            'min_cls_batch_std': min(r['feature_stat']['cls_batch_std_mean'] for r in steps),
            'min_patch_spatial_std': min(r['feature_stat']['patch_spatial_std_mean'] for r in steps),
            'final_parameter_delta_l2': steps[-1]['tracked_parameter_delta_l2'],
            'min_grad_norm': min(r['grad_norm'] for r in steps),
            'peak_gpu_bytes': max(r['peak_gpu_bytes'] for r in steps),
            'mean_seconds_per_step': steps[-1]['seconds']/100,
            'loss_range': [min(r['loss'] for r in steps), max(r['loss'] for r in steps)]}


def main():
    if (BASE/'STOP.json').exists():
        raise SystemExit('Existing STOP marker requires investigation')
    guard = Guard()
    status_file = BASE/'stability_batch_status.json'
    try:
        guard.start()
        save(status_file, {'status': 'running', 'assignments': ASSIGNMENTS, 'started': time.time()})
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(guard.job, f'{arm}_seed0_batch16_stability100', gpu,
                                   'train_ssl.py', ['--arm', arm, '--batch', '16',
                                                    '--steps', '100', '--seed', '0']): arm
                       for arm, gpu in ASSIGNMENTS}
            paths = {futures[future]: future.result() for future in as_completed(futures)}
        rows = [assess(paths[arm]) for arm, _ in ASSIGNMENTS]
        b = json.loads((BASE/'jobs/B_seed0_batch16_stability100/result.json').read_text())
        assert all(row['sample_chain'] == b['sample_chain'] for row in rows)
        save(status_file, {'status': 'completed', 'assignments': ASSIGNMENTS,
                           'result': rows, 'ended': time.time()})
        print(json.dumps(rows))
        return 0
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(status_file, {'status': 'blocked', 'reason': repr(exc), 'ended': time.time()})
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    raise SystemExit(main())
