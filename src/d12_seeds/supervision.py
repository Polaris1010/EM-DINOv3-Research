"""Temporary monitoring handover; never signals SSL ranks or changes their training state."""
import os
import signal
import subprocess
import time
import psutil
from common import R, NR, atomic, read, heartbeat

def identity(pid, created):
    try:
        p = psutil.Process(pid)
        return p if p.create_time() == created else None
    except psutil.NoSuchProcess:
        return None

def acquire():
    pid = int(subprocess.check_output(['systemctl', 'show', 'coverage-nr-p0-20261021.service', '-p', 'MainPID', '--value'], text=True))
    p = psutil.Process(pid)
    assert str(NR / 'code/run.py') in p.cmdline()
    assert read(NR / 'STATUS.json')['phase'].startswith('SSL_')
    receipt = {'controller_pid': pid, 'created': p.create_time(), 'acquired_at': time.time(),
               'rank_processes': {str(c.pid): c.create_time() for c in p.children(recursive=True)},
               'training_paused': False, 'supervisor_only_suspended': True, 'released': False}
    atomic(R / 'supervision.json', receipt)
    # Stop ONLY the Python supervisor. Its torchrun and three ranks continue unchanged.
    p.send_signal(signal.SIGSTOP)
    deadline = time.monotonic() + 5
    while p.status() != psutil.STATUS_STOPPED:
        assert time.monotonic() < deadline
        time.sleep(.05)
    heartbeat(NR / 'HEARTBEAT')

def maintain():
    j = read(R / 'supervision.json')
    p = identity(j['controller_pid'], j['created'])
    assert p and p.status() == psutil.STATUS_STOPPED, 'Original supervisor identity/status changed'
    heartbeat(NR / 'HEARTBEAT')
    for pid, created in j['rank_processes'].items():
        q = identity(int(pid), created)
        if q:
            assert q.status() != psutil.STATUS_STOPPED, 'SSL process must continue'
    assert time.time() - j['acquired_at'] < 36 * 3600, 'Bounded monitoring handover'

def release():
    path = R / 'supervision.json'
    if not path.exists():
        return
    j = read(path)
    if j['released']:
        return
    p = identity(j['controller_pid'], j['created'])
    if p:
        heartbeat(NR / 'HEARTBEAT')
        p.send_signal(signal.SIGCONT)
    j.update(released=True, released_at=time.time())
    atomic(path, j)

if __name__ == '__main__':
    # systemd ExecStopPost runs after this service's whole cgroup has terminated.
    release()
