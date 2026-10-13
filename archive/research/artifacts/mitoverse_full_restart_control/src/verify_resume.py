"""Verify that a fresh-run checkpoint resumes the exact image sample sequence."""

import json
from pathlib import Path

from full_manifest import sha256

RUN = Path(__file__).resolve().parent.parent
JOBS = RUN / 'src/jobs'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    original = JOBS / 'B_seed0_batch16_smoke10'
    resumed = JOBS / 'B_seed0_batch16_resume20'
    continuous = JOBS / 'B_seed0_batch16_smoke20'
    receipt = load(original / 'latest_checkpoint.json')
    checkpoint = Path(receipt['path'])
    if checkpoint != original / 'checkpoint_step10.pth' or sha256(checkpoint) != receipt['sha256']:
        raise ValueError('The step-10 checkpoint receipt is invalid')
    a, b = load(resumed / 'result.json'), load(continuous / 'result.json')
    if (a['status'] != 'complete' or b['status'] != 'complete'
            or a['completed_steps'] != 20 or b['completed_steps'] != 20
            or a['sample_chain'] != b['sample_chain']):
        raise ValueError('Resumed and continuous sample sequences differ')
    resumed_steps = [json.loads(s) for s in (resumed / 'steps.jsonl').read_text().splitlines()]
    full_steps = [json.loads(s) for s in (continuous / 'steps.jsonl').read_text().splitlines()]
    if ([s['step'] for s in resumed_steps] != list(range(11, 21))
            or [s['step'] for s in full_steps] != list(range(1, 21))
            or any(x['sample_chain'] != y['sample_chain']
                   for x, y in zip(resumed_steps, full_steps[10:]))):
        raise ValueError('Resumed per-step sample chains differ')
    evidence = {'status': 'passed', 'checkpoint_path': str(checkpoint),
                'checkpoint_sha256': receipt['sha256'], 'resume_start_step': 10,
                'target_step': 20, 'sample_chain': a['sample_chain'],
                'comparison': 'step-10 resumed and step-0 continuous B sample chains match'}
    (RUN / 'evidence/resume_sequence_check.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
