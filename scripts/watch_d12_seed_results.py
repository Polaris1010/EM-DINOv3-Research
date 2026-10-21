"""Finite passive delivery of the eight authorized head runs; starts no experiments."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.pop('SSLKEYLOGFILE', None)
os.environ.update(PYTHONUTF8='1', GIT_CONFIG_COUNT='3', GIT_CONFIG_KEY_0='credential.helper', GIT_CONFIG_VALUE_0='',
                 GIT_CONFIG_KEY_1='credential.helper',
                 GIT_CONFIG_VALUE_1='!python E:/mira_website_demo/artifacts/d12_control_20261017/win_git_credential.py',
                 GIT_CONFIG_KEY_2='http.sslBackend', GIT_CONFIG_VALUE_2='schannel')
import watch_d12_100k_results as base

ROOT = base.ROOT
base.SCRATCH = ROOT.parent / 'artifacts/d12_three_seeds_20261022'
base.REMOTE = '/opt/data/3_117/tanghr/codex/d12_three_seeds_20261022'
base.BRANCH = 'codex/d12-three-seed-validation'

def collect(sftp, status):
    folder = base.SCRATCH / 'terminal_collected'
    folder.mkdir(exist_ok=True)
    names = ['STATUS.json', 'supervision.json', 'protocol.json', 'D12_three_seed_report.md']
    names += [v.filename for v in sftp.listdir_attr(base.REMOTE)
              if v.filename.startswith('D12_three_seed_') and v.filename.endswith(('.json', '.csv'))
              and v.filename != 'D12_three_seed_MB1_per_image.csv']
    for model in ['D1-30k', 'D2-30k', 'D1-100k', 'D2-100k']:
        for seed in [20261015, 20261016]:
            names += [f'downstream/runs/{model}/seed{seed}/{v}.json'
                      for v in ['training_complete', 'inference_complete', 'head_step040000', 'run_protocol', 'curve']]
    names += ['evidence/' + v.filename for v in sftp.listdir_attr(base.REMOTE + '/evidence')
              if v.filename.endswith('.json')]
    for name in sorted(set(names)):
        try:
            st = sftp.stat(base.REMOTE + '/' + name)
        except OSError:
            continue
        assert st.st_size < 2_000_000, name
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(base.REMOTE + '/' + name, str(path))
    return folder

def publish(folder, status, api):
    # The unrelated scratch script existed before this task and is never staged.
    dirty = base.git('status', '--porcelain')
    if base.git('branch', '--show-current') != base.BRANCH or dirty not in ['', '?? scripts/phase0_complementarity.py']:
        return {'status': 'COLLECTED_CHECKOUT_CHANGED_OR_DIRTY', 'folder': str(folder), 'server_status': status}
    before = json.loads((ROOT / 'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:59]
    for path in folder.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(folder)
        target = ROOT / 'docs/D12_three_seed_report.md' if path.name == 'D12_three_seed_report.md' else ROOT / 'experiments/D12-three-seeds' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ip = ROOT / 'experiments/D12-three-seeds/run_index.json'
    index = json.loads(ip.read_text(encoding='utf-8'))
    index.update(status='VERIFIED_COMPLETED' if status['status'] == 'complete' else 'VERIFIED_FAILED',
                 status_evidence='experiments/D12-three-seeds/STATUS.json')
    for run in index['runs']:
        d = ROOT / 'experiments/D12-three-seeds/downstream/runs' / run['model'] / f"seed{run['seed']}"
        complete = d / 'training_complete.json'
        if complete.exists():
            j = json.loads(complete.read_text(encoding='utf-8'))
            run.update(completed_steps=j['completed_steps'], status='VERIFIED_COMPLETED')
            meta = json.loads((d / 'head_step040000.json').read_text(encoding='utf-8'))
            run.update(checkpoint_sha256=meta['sha256'])
        elif run['status'] == 'VERIFIED_RUNNING':
            run['status'] = 'VERIFIED_FAILED'
    if status['status'] == 'complete':
        index.update(metrics_path='experiments/D12-three-seeds/D12_three_seed_summary.csv', report_path='docs/D12_three_seed_report.md')
    ip.write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8', newline='\n')
    note = 'D1/D2三个decoder seed任务终态：' + status['status'] + '，UTC ' + datetime.datetime.now(datetime.timezone.utc).isoformat() + '。'
    note += ' 详见docs/D12_three_seed_report.md与本轮STATUS；固定划分、四个已有encoder，没有新SSL。' if status['status'] == 'complete' else ' 保留阶段证据与已完成产物，不自动重启或改参。'
    note += ' 本轮结束停止。Coverage-NR是独立已授权服务，按其既有任务继续，不重复启动。'
    for name in ['AGENTS.md', 'START_HERE.md', 'docs/PROJECT_STATE.md', 'docs/NEW_WINDOW_HANDOFF.md', 'docs/NEXT_ACTION.md']:
        p = ROOT / name
        p.write_text('> ' + note + '\n\n' + p.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    p = ROOT / 'docs/work_logs/20261022_29_d12_three_seed_start.md'
    p.write_text(p.read_text(encoding='utf-8') + '\n## 终态收取\n\n' + note + '\n', encoding='utf-8', newline='\n')
    p = ROOT / 'docs/D12_three_seed_PR.md'
    p.write_text(note + '\n\n' + p.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    for args in [['scripts/update_registry.py'], ['scripts/update_registry.py', '--check'],
                 ['scripts/validate_repository.py'], ['-m', 'unittest', 'discover', '-s', 'tests', '-q']]:
        subprocess.run([sys.executable, *args], cwd=ROOT, env=base.ENV, check=True)
    assert json.loads((ROOT / 'experiments/registry.yaml').read_text(encoding='utf-8'))['experiments'][:59] == before
    base.git('add', 'AGENTS.md', 'START_HERE.md', 'docs/D12_three_seed_report.md' if status['status'] == 'complete' else 'docs/D12_three_seed_execution.md',
             'docs/PROJECT_STATE.md', 'docs/NEW_WINDOW_HANDOFF.md', 'docs/NEXT_ACTION.md',
             'docs/D12_three_seed_PR.md', 'docs/work_logs/20261022_29_d12_three_seed_start.md',
             'experiments/D12-three-seeds', 'experiments/registry.yaml')
    base.git('commit', '-m', 'Collect D1 D2 three-seed metrics and final research comparison')
    base.git('-c', 'http.sslBackend=schannel', '-c', 'http.version=HTTP/1.1', 'push', 'origin', base.BRANCH)
    head = base.git('rev-parse', 'HEAD')
    prs = api('/repos/Polaris1010/EM-DINOv3-Research/pulls?state=open&head=Polaris1010:' + base.BRANCH)
    assert prs
    pr = api('/repos/Polaris1010/EM-DINOv3-Research/pulls/' + str(prs[0]['number']), 'PATCH',
             {'body': (ROOT / 'docs/D12_three_seed_PR.md').read_text(encoding='utf-8')})
    assert pr['draft'] and not pr['merged'] and pr['head']['sha'] == head
    return {'status': 'PUSHED', 'commit': head, 'pr': pr['html_url'], 'server_status': status['status'], 'starts_experiments': False}

base.collect, base.publish = collect, publish

if __name__ == '__main__':
    try:
        base.main()
    except BaseException as e:
        base.SCRATCH.mkdir(exist_ok=True)
        (base.SCRATCH / 'collector_error.json').write_text(json.dumps({'error': repr(e), 'starts_experiments': False}) + '\n')
        raise
