"""Resume only missing distance metrics after a controlled scheduler/backend change."""
import fcntl
import signal
import time
import psutil
from common import R, atomic, read
from run import Controller
import supervision

def finish(ctl):
    assert read(R / 'GPU_SCORING.json')['enabled']
    allpaths = list(R.glob('downstream/runs/*/seed*/*_overlap.json'))
    assert len(allpaths) == 304
    pending = sorted([p for p in allpaths if not p.with_name(p.name.replace('_overlap.json', '_metrics.json')).exists()],
                     key=lambda p: read(p)['valid_voxels'])
    assert pending, 'Already complete: do not repeat scoring'
    supervision.acquire()
    ctl.phase = 'exact_GPU_distance_metrics'
    active = []
    while pending or active:
        for p, need in list(active):
            if p.poll() is not None:
                assert p.returncode == 0, f'GPU scoring failed:{p.pid}'
                active.remove((p, need))
        while pending and len(active) < 3:
            need = 45 * read(pending[0])['valid_voxels'] + (4 << 30)
            if need > psutil.virtual_memory().available-(32 << 30)-sum(n for _,n in active):
                assert active, 'Insufficient RAM for exact metrics'
                break
            path = pending.pop(0)
            p = ctl.launch([R / 'code/downstream.py', 'score', '--path', path],
                           'GPU_score_'+path.parent.parent.name+'_'+path.parent.name+'_'+path.stem)
            active.append((p, need))
        ctl.health()
        time.sleep(5)
    assert len(list(R.glob('downstream/runs/*/seed*/*_metrics.json'))) == 304
    supervision.release()
    ctl.phase = 'three_seed_summary'
    ctl.wait([ctl.launch([R / 'code/report.py'], 'GPU_completed_report')])
    atomic(R / 'STATUS.json', {'status': 'complete', 'new_head_runs': 8, 'steps_each': 40000,
           'total_decoder_seeds': 3, 'fixed_split': True, 'new_SSL_training': False,
           'development_rows': 456, 'MB1_image_rows': 31140, 'gpu0_used': False, 'website_http': 200,
           'report': 'D12_three_seed_report.md', 'remaining_scoring_accelerated': True,
           'further_experiments_started': False})

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('service stop')))
    with (R / 'RUN.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ctl = Controller()
        try:
            finish(ctl)
        except BaseException as e:
            atomic(R / 'STATUS.json', {'status': 'failed', 'phase': ctl.phase, 'error': repr(e)})
            raise
        finally:
            ctl.cleanup()
