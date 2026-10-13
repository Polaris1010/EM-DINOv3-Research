"""Run the remaining three 10-step SSL short checks under one GPU guard."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

from guard import Guard, save
from settings import BASE

ASSIGNMENTS = [('L', 1), ('T', 2), ('LT', 3)]


def main() -> int:
    if (BASE / 'STOP.json').exists():
        raise SystemExit('Existing STOP marker requires investigation')
    guard = Guard()
    status = BASE / 'smoke_batch_status.json'
    completed = []
    try:
        guard.start()
        save(status, {'status': 'running', 'assignments': ASSIGNMENTS, 'started': time.time()})
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(guard.job, f'{arm}_seed0_batch16_smoke10', gpu,
                            'train_ssl.py', ['--arm', arm, '--batch', '16', '--steps', '10', '--seed', '0']): arm
                for arm, gpu in ASSIGNMENTS
            }
            for future in as_completed(futures):
                arm = futures[future]
                path = future.result()
                result = json.loads((path / 'result.json').read_text())
                if result['status'] != 'complete' or result['completed_steps'] != 10:
                    raise RuntimeError(f'{arm} short check incomplete')
                completed.append({'arm': arm, 'sample_chain': result['sample_chain'],
                                  'step': result['completed_steps']})
                save(status, {'status': 'running', 'completed': completed, 'updated': time.time()})
        b = json.loads((BASE / 'jobs/B_seed0_batch16_smoke10/result.json').read_text())
        if any(row['sample_chain'] != b['sample_chain'] for row in completed):
            raise RuntimeError('Four arms consumed different image sample chains')
        save(status, {'status': 'completed', 'completed': completed, 'ended': time.time()})
        print(json.dumps(completed))
        return 0
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(status, {'status': 'blocked', 'completed': completed,
                      'reason': repr(exc), 'ended': time.time()})
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    raise SystemExit(main())
