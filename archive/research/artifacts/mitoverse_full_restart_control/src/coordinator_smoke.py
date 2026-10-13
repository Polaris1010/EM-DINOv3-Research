"""Dispatch one guarded B/L/T/LT smoke job on authorized physical GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from guard import Guard, save
from settings import BASE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['B', 'L', 'T', 'LT'], required=True)
    parser.add_argument('--gpu', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--batch', type=int, choices=[4, 8, 16], default=16)
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--steps', type=int, choices=[5, 10, 20, 100], default=10)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if (BASE / 'STOP.json').exists():
        raise SystemExit('Existing STOP marker requires investigation')
    guard = Guard()
    kind = 'stability' if args.steps == 100 else 'resume' if args.resume else 'smoke'
    status_file = BASE / f'{args.arm}_{kind}_status.json'
    try:
        guard.start()
        save(status_file, {'status': 'running', 'arm': args.arm, 'gpu': args.gpu,
                           'batch': args.batch, 'started': time.time()})
        suffix = f'_attempt{args.attempt}' if args.attempt else ''
        name = f'{args.arm}_seed0_batch{args.batch}_{kind}{args.steps}{suffix}'
        train_args = ['--arm', args.arm, '--batch', str(args.batch), '--steps', str(args.steps), '--seed', '0']
        if args.resume:
            train_args.extend(['--resume', str(args.resume)])
        path = guard.job(name, args.gpu, 'train_ssl.py', train_args)
        result = json.loads((path / 'result.json').read_text())
        save(status_file, {'status': 'completed', 'arm': args.arm, 'gpu': args.gpu,
                           'batch': args.batch, 'result': result, 'ended': time.time()})
        print(json.dumps(result))
        return 0
    except Exception as exc:
        guard.stop_all(repr(exc))
        save(status_file, {'status': 'blocked', 'reason': repr(exc), 'ended': time.time()})
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    raise SystemExit(main())
