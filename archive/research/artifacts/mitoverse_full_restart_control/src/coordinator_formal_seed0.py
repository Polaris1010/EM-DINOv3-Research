"""Persistent guarded SSL seed-0 queue: B/L/T, then LT on the first free GPU."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import time

from guard import Guard, save
from settings import BASE

INITIAL = [('B', 1), ('L', 2), ('T', 3)]
TARGET = 100000
STATUS = BASE/'formal_seed0_status.json'


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def find_next(arm):
    parts = sorted((BASE/'jobs').glob(f'formal_seed0_{arm}_part*'),
                   key=lambda p: int(p.name.rsplit('part', 1)[1]))
    if not parts:
        return f'formal_seed0_{arm}_part0', None
    last = parts[-1]
    result_path = last/'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get('status') == 'complete' and result.get('completed_steps') == TARGET:
            return None, result
    receipt = last/'latest_checkpoint.json'
    if not receipt.exists():
        raise RuntimeError(f'{arm} has an incomplete part without an atomic checkpoint: {last}')
    checkpoint = json.loads(receipt.read_text())
    path = last/f"checkpoint_step{checkpoint['step']}.pth"
    if path != __import__('pathlib').Path(checkpoint['path']) or not path.exists():
        raise RuntimeError(f'{arm} latest checkpoint receipt/path mismatch')
    if file_sha(path) != checkpoint['sha256']:
        raise RuntimeError(f'{arm} latest checkpoint hash mismatch')
    number = int(last.name.rsplit('part', 1)[1]) + 1
    return f'formal_seed0_{arm}_part{number}', path


def run_arm(guard, arm, gpu):
    name, value = find_next(arm)
    if name is None:
        return {'arm': arm, 'gpu': gpu, 'status': 'already_complete', 'result': value}
    args = ['--arm', arm, '--batch', '16', '--steps', str(TARGET), '--seed', '0']
    if value is not None:
        args += ['--resume', str(value)]
    path = guard.job(name, gpu, 'train_ssl.py', args)
    result = json.loads((path/'result.json').read_text())
    if result['status'] != 'complete' or result['completed_steps'] != TARGET:
        raise RuntimeError(f'{arm} result incomplete')
    for step in (10000, 30000, 60000, 100000):
        if not (path/f'ema_step{step}.json').exists():
            # Earlier exports may belong to a preceding resumed part.
            if not any((p/f'ema_step{step}.json').exists() for p in (BASE/'jobs').glob(f'formal_seed0_{arm}_part*')):
                raise RuntimeError(f'{arm} missing EMA milestone {step}')
    return {'arm': arm, 'gpu': gpu, 'status': 'completed', 'job': name,
            'completed_steps': TARGET, 'sample_chain': result['sample_chain']}


def main():
    if (BASE/'STOP.json').exists():
        raise SystemExit('STOP marker blocks formal dispatch')
    admission = json.loads((BASE.parent/'evidence/formal_seed0_admission.json').read_text())
    if (admission['status'] != 'admitted' or admission['batch'] != 16
            or admission['data_scope'] != 'all_mitoverse'
            or admission['initialization'] != 'official_natural_vitl16'
            or admission['resume'] is not None):
        raise SystemExit('Formal admission receipt is invalid')
    for name, expected in admission['method_hashes'].items():
        path = (BASE.parent/'topology_calibration'/name if name == 'topology_selected_locked.json'
                else BASE.parent/'data'/name if name == 'pretrain_all_mitoverse.json'
                else BASE/name)
        if file_sha(path) != expected:
            raise SystemExit(f'Formal source hash changed: {name}')
    previous = json.loads(STATUS.read_text()) if STATUS.exists() else None
    if previous and previous.get('status') == 'completed':
        print(json.dumps(previous))
        return 0
    guard = Guard()
    completed = []
    try:
        guard.start()
        save(STATUS, {'status': 'running', 'target_updates_each': TARGET,
                      'initial_assignments': INITIAL, 'queued': ['LT'],
                      'started': time.time(), 'completed': completed})
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(run_arm, guard, arm, gpu): (arm, gpu) for arm, gpu in INITIAL}
            lt_dispatched = False
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in ready:
                    arm, gpu = futures.pop(future)
                    row = future.result()
                    completed.append(row)
                    if not lt_dispatched:
                        futures[pool.submit(run_arm, guard, 'LT', gpu)] = ('LT', gpu)
                        lt_dispatched = True
                    save(STATUS, {'status': 'running', 'target_updates_each': TARGET,
                                  'completed': completed,
                                  'active_assignments': list(futures.values()),
                                  'last_updated': time.time()})
        if {row['arm'] for row in completed} != {'B', 'L', 'T', 'LT'}:
            raise RuntimeError('Four-arm completion set mismatch')
        save(STATUS, {'status': 'completed', 'completed': completed, 'ended': time.time()})
        print(json.dumps(completed))
        return 0
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(STATUS, {'status': 'blocked', 'completed': completed,
                      'reason': repr(exc), 'ended': time.time()})
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    raise SystemExit(main())
