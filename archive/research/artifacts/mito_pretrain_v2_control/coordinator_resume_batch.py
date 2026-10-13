"""Guarded 5+5 recovery gate for L, T and LT on physical GPUs 1–3."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import time

from guard import Guard, save
from settings import BASE, PYTHON

ASSIGNMENTS = [('L', 1), ('T', 2), ('LT', 3)]


def one(guard, arm, gpu):
    base_args = ['--arm', arm, '--batch', '16', '--seed', '0']
    first = guard.job(f'{arm}_seed0_batch16_smoke5', gpu, 'train_ssl.py',
                      [*base_args, '--steps', '5'])
    second = guard.job(f'{arm}_seed0_batch16_resume10', gpu, 'train_ssl.py',
                       [*base_args, '--steps', '10', '--resume',
                        str(first/'checkpoint_step5.pth')])
    return second


def main():
    if (BASE/'STOP.json').exists():
        raise SystemExit('Existing STOP marker requires investigation')
    guard = Guard()
    status_file = BASE/'resume_batch_status.json'
    try:
        guard.start()
        save(status_file, {'status': 'running', 'assignments': ASSIGNMENTS, 'started': time.time()})
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(one, guard, arm, gpu): arm for arm, gpu in ASSIGNMENTS}
            paths = {futures[future]: future.result() for future in as_completed(futures)}
        rows = []
        for arm, _ in ASSIGNMENTS:
            original = BASE/f'jobs/{arm}_seed0_batch16_smoke10'
            if arm == 'T':
                original = BASE/'jobs/T_seed0_batch16_smoke10_attempt1'
            command = [str(PYTHON), str(BASE/'verify_ssl_resume.py'),
                       '--continuous', str(original/'checkpoint_step10.pth'),
                       '--resumed', str(paths[arm]/'checkpoint_step10.pth'),
                       '--tolerance', '5e-5']
            checked = subprocess.run(command, text=True, capture_output=True, check=True)
            row = json.loads(checked.stdout)
            assert row['numerically_consistent'] and row['sample_chain_equal']
            rows.append({'arm': arm, **row})
        save(status_file, {'status': 'completed', 'result': rows, 'ended': time.time()})
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
