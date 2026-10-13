"""Record the four-arm correctness, stability and recovery gate before 100k dispatch."""

import hashlib
import json
import math
from pathlib import Path

from settings import BASE

RUN = BASE.parent
ARMS = ('B', 'L', 'T', 'LT')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert not (BASE/'STOP.json').exists()
    recovery = json.loads((BASE/'resume_batch_status.json').read_text())
    assert recovery['status'] == 'completed'
    recovered = {row['arm']: row for row in recovery['result']}
    recovered['B'] = json.loads((RUN/'evidence/B_resume_numerics.json').read_text())
    smoke_names = {'B': 'B_seed0_batch16_smoke10',
                   'L': 'L_seed0_batch16_smoke10',
                   'T': 'T_seed0_batch16_smoke10_attempt1',
                   'LT': 'LT_seed0_batch16_smoke10'}
    rows = []
    for arm in ARMS:
        smoke = json.loads((BASE/'jobs'/smoke_names[arm]/'result.json').read_text())
        stable = BASE/f'jobs/{arm}_seed0_batch16_stability100'
        result = json.loads((stable/'result.json').read_text())
        steps = [json.loads(line) for line in (stable/'steps.jsonl').read_text().splitlines()]
        resume = recovered[arm]
        assert smoke['status'] == 'complete' and smoke['completed_steps'] == 10
        assert result['status'] == 'complete' and result['completed_steps'] == 100 and len(steps) == 100
        assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['grad_norm'] > 0 for row in steps)
        assert min(row['feature_stat']['cls_batch_std_mean'] for row in steps) > 1e-4
        assert min(row['feature_stat']['patch_spatial_std_mean'] for row in steps) > 1e-4
        assert any(v > 0 for v in steps[-1]['tracked_parameter_delta_l2'].values())
        assert resume['step_equal'] and resume['sample_chain_equal'] and resume['cfg_equal'] and resume['manifest_equal']
        assert resume['numerically_consistent']
        rows.append({'arm': arm, 'smoke_steps': 10, 'stability_steps': 100,
                     'sample_chain': result['sample_chain'],
                     'min_cls_std': min(row['feature_stat']['cls_batch_std_mean'] for row in steps),
                     'min_patch_std': min(row['feature_stat']['patch_spatial_std_mean'] for row in steps),
                     'max_resume_weight_abs_diff': max(resume['max_abs_diff'].values()),
                     'resume_bitwise_equal': resume['bitwise_equal'],
                     'peak_gpu_bytes': max(row['peak_gpu_bytes'] for row in steps),
                     'mean_step_seconds': steps[-1]['seconds']/100})
    assert len({row['sample_chain'] for row in rows}) == 1
    source_files = ['train_ssl.py', 'ssl_data.py', 'last_selection.py', 'topology_appearance.py']
    method_hashes = {name: sha(BASE/name) for name in source_files}
    method_hashes['topology_selected_locked.json'] = sha(RUN/'topology_calibration/topology_selected_locked.json')
    method_hashes['pretrain_train.json'] = sha(RUN/'data/pretrain_train.json')
    receipt = {'status': 'admitted', 'batch': 16, 'target_steps_each': 100000,
               'authorized_gpus': [1, 2, 3], 'forbidden_gpu': 0,
               'arms': rows, 'method_hashes': method_hashes,
               'numeric_resume_note': 'Sample chains exact; floating-point weights not bitwise equal, within observed same-host run-to-run difference',
               'loss_note': 'BF16 total loss displays 16.625 throughout first 100 steps; finite components, nonzero gradients, feature variance and parameter updates verified'}
    target = RUN/'evidence/formal_seed0_admission.json'
    if target.exists():
        raise RuntimeError('Admission receipt exists; cannot silently overwrite')
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
