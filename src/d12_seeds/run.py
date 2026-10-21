"""Eight additional40k readouts, fixed development/MB1 scoring, three-seed report, then stop."""
import csv
import fcntl
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
import psutil
from common import R, NR, PY, atomic, read, sha
import supervision

class Controller:
    def __init__(self):
        self.children, self.files = [], []
        self.phase = 'initialize'
        self.start = time.monotonic()
        self.gpus = {int(r[0]): r[1].strip() for r in csv.reader(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
        self.baseline = {p.pid: p.create_time() for p in psutil.process_iter()}
        self.protocol = read(R / 'protocol.json')
        self.models = list(self.protocol['models'])

    def launch(self, args, name, gpu=None):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': self.gpus[gpu] if gpu else '',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'PYTHONDONTWRITEBYTECODE': '1', 'MV2_GPU_EDT': '0',
               'NUMBA_CACHE_DIR': str(R / 'cache/numba'), 'CUPY_CACHE_DIR': str(R / 'cache/cupy'),
               'CUDA_CACHE_PATH': str(R / 'cache/cuda'), 'PYTORCH_KERNEL_CACHE_PATH': str(R / 'cache/pytorch')}
        f = (R / 'logs' / (name + '.log')).open('a')
        p = subprocess.Popen([str(PY), *map(str, args)], env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(p)
        self.files.append(f)
        return p

    def health(self):
        assert time.monotonic() - self.start < 48 * 3600
        if (R / 'supervision.json').exists() and not read(R / 'supervision.json')['released']:
            supervision.maintain()
        assert urllib.request.urlopen('http://127.0.0.1:7860/', timeout=10).status == 200
        roots = {p.pid for p in self.children}
        if (R / 'supervision.json').exists():
            j = read(R / 'supervision.json')
            if supervision.identity(j['controller_pid'], j['created']):
                roots.add(j['controller_pid'])
        for row in csv.reader(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines()):
            if row[0].strip() not in {self.gpus[g] for g in [1, 2, 3]}:
                continue
            try:
                p = psutil.Process(int(row[1]))
                assert self.baseline.get(p.pid) == p.create_time() or p.pid in roots or any(v.pid in roots for v in p.parents()), 'Unexpected concurrent GPU task'
            except psutil.NoSuchProcess:
                pass
        progress = {}
        for p in R.glob('downstream/runs/*/seed*/progress.json'):
            progress[str(p.relative_to(R))] = read(p)
        nr = NR / 'runs/Coverage-NR/formal/progress.json'
        atomic(R / 'STATUS.json', {'status': 'running', 'phase': self.phase, 'seconds': time.monotonic()-self.start,
               'progress': progress, 'active_pids': [p.pid for p in self.children if p.poll() is None],
               'Coverage_NR_progress': read(nr) if nr.exists() else None, 'gpu0_used': False, 'website_http': 200})

    def wait(self, jobs):
        while True:
            self.health()
            assert all(p.poll() in [None, 0] for p in jobs), 'Own child failed; inspect logs'
            if all(p.poll() == 0 for p in jobs):
                return
            time.sleep(5)

    def gpu_phase(self, mode):
        pending = [(m, s) for s in self.protocol['new_seeds'] for m in self.models]
        active = {}
        self.phase = mode
        while pending or active:
            for g, p in list(active.items()):
                if p.poll() is not None:
                    assert p.returncode == 0, f'{mode} child failed:{p.pid}'
                    del active[g]
            for g in [1, 2, 3]:
                if g in active or not pending:
                    continue
                m, s = pending.pop(0)
                script = 'benchmark.py' if mode == 'benchmark' else 'downstream.py'
                args = [R / 'code' / script]
                if mode != 'benchmark':
                    args.append(mode)
                args += ['--model', m, '--seed', s]
                active[g] = self.launch(args, f'{mode}_{m}_{s}', g)
            self.health()
            time.sleep(5)

    def run(self):
        # Full byte hashes, existing completed first seeds, identical fixed inputs.
        for m, spec in self.protocol['models'].items():
            assert sha(spec['encoder_path']) == spec['encoder_sha256']
            old = Path(spec['existing_head_dir'])
            j = read(old / 'training_complete.json')
            assert j['completed_steps'] == 40000 and j['encoder_unchanged']
            assert j['encoder_sha256'] == spec['encoder_sha256']
            assert sha(old / 'head_step040000.pth') == spec['existing_head_sha256']
            assert len(list(old.glob('*_metrics.json'))) == 38
        supervision.acquire()
        self.phase = 'shared_VRAM_smoke'
        self.wait([self.launch([R / 'code/downstream.py', 'smoke', '--model', self.models[0], '--seed', '20261015'], 'smoke', 3)])
        self.gpu_phase('train')
        # Matching initialization and sampling across all four encoders for each seed.
        for s in self.protocol['new_seeds']:
            rows = [read(R / 'downstream/runs' / m / f'seed{s}/training_complete.json') for m in self.models]
            assert all(v['completed_steps'] == 40000 and v['encoder_unchanged'] for v in rows)
            assert len({v['decoder_initial_sha256'] for v in rows}) == 1
            assert len({v['sampling_hash'] for v in rows}) == 1
        self.gpu_phase('infer')
        self.gpu_phase('benchmark')
        # No further CUDA children: let the original supervisor continue its own pipeline.
        supervision.release()
        self.phase = 'exact_CPU_distance_metrics'
        pending = sorted(R.glob('downstream/runs/*/seed*/*_overlap.json'), key=lambda p: read(p)['valid_voxels'])
        assert len(pending) == 304
        active = []
        while pending or active:
            for p, need in list(active):
                if p.poll() is not None:
                    assert p.returncode == 0, f'CPU scoring failed:{p.pid}'
                    active.remove((p, need))
            while pending and len(active) < 3:
                need = 45 * read(pending[0])['valid_voxels'] + (4 << 30)
                if need > psutil.virtual_memory().available - (32 << 30) - sum(n for _, n in active):
                    assert active, 'Insufficient RAM for exact evaluator'
                    break
                path = pending.pop(0)
                p = self.launch([R / 'code/downstream.py', 'score', '--path', path], 'score_' + path.parent.parent.name + '_' + path.parent.name + '_' + path.stem)
                active.append((p, need))
            self.health()
            time.sleep(5)
        self.phase = 'three_seed_summary'
        self.wait([self.launch([R / 'code/report.py'], 'report')])
        atomic(R / 'STATUS.json', {'status': 'complete', 'new_head_runs': 8, 'steps_each': 40000,
               'total_decoder_seeds': 3, 'fixed_split': True, 'new_SSL_training': False,
               'development_rows': 456, 'MB1_image_rows': 31140, 'gpu0_used': False,
               'website_http': 200, 'report': 'D12_three_seed_report.md', 'further_experiments_started': False})

    def cleanup(self):
        for p in self.children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in self.children:
            if p.poll() is None:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        # Must resume NR only AFTER every new CUDA process has exited.
        supervision.release()
        for f in self.files:
            f.close()

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('service stop')))
    with (R / 'RUN.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ctl = Controller()
        try:
            ctl.run()
        except BaseException as e:
            atomic(R / 'STATUS.json', {'status': 'failed', 'phase': ctl.phase, 'error': repr(e)})
            raise
        finally:
            ctl.cleanup()
