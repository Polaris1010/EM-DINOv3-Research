"""Own-process B_full launcher and 60-second GPU/website resource guard."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil


RUN = Path(__file__).resolve().parent.parent
OUT = RUN / 'runs' / 'B_full' / 'formal_attempt1'
STATUS = RUN / 'STATUS.json'
HEARTBEAT = RUN / 'audit' / 'gpu_guard_heartbeat.txt'
LOG = RUN / 'audit' / 'gpu_guard.jsonl'
TRAIN_LOG = RUN / 'runs' / 'B_full' / 'formal_attempt1_train.log'
PYTHON = '/opt/data/3_117/tanghr/dinov3_shared/bin/python'
DINO = '/opt/data/3_117/tanghr/DINOv3/dinov3'
ZARR_DEPS = '/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps'
WEIGHT = '/opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu_inventory():
    gpu = {}
    for row in csv.reader(run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                               '--format=csv,noheader,nounits']).splitlines()):
        i = int(row[0].strip())
        gpu[i] = {'uuid': row[1].strip(), 'total_mib': int(row[2]), 'used_mib': int(row[3]),
                  'free_mib': int(row[4]), 'utilization': int(row[5]), 'temperature': int(row[6])}
    return gpu


def gpu_apps(gpu):
    uuid_to_index = {row['uuid']: i for i, row in gpu.items()}
    apps = []
    raw = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
    for row in csv.reader(raw.splitlines()):
        if len(row) < 3 or row[0].strip() not in uuid_to_index:
            continue
        try:
            pid = int(row[1])
            process = psutil.Process(pid)
            birth = process.create_time()
            command = process.cmdline()
        except (ValueError, psutil.Error):
            continue
        apps.append({'gpu': uuid_to_index[row[0].strip()], 'pid': pid,
                     'birth': birth, 'memory_mib': int(row[2]), 'command': command})
    return apps


def website_ok():
    try:
        with urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_heartbeat():
    temporary = HEARTBEAT.with_suffix('.tmp')
    temporary.write_text(str(time.time()) + '\n')
    temporary.replace(HEARTBEAT)


def is_own_process(pid: int, leader: int):
    try:
        process = psutil.Process(pid)
        return process.pid == leader or any(parent.pid == leader for parent in process.parents())
    except psutil.Error:
        return False


def main() -> int:
    admission = json.loads((RUN / 'audit/formal_B_full_admission.json').read_text())
    if admission['status'] != 'admitted' or any(value != 'PASS' for value in admission['gates'].values()):
        raise SystemExit('Formal B_full admission is incomplete')
    for path, key in [(RUN / 'scripts/train_B_full.py', 'trainer_sha256'),
                      (RUN / 'scripts/ssl_full_data.py', 'sampler_sha256'),
                      (RUN / 'configs/B_full.yaml', 'config_sha256'),
                      (RUN / 'data/pretrain_all_mitoverse.json', 'manifest_sha256')]:
        if digest(path) != admission[key]:
            raise SystemExit(f'Formal input changed after admission: {path}')
    if OUT.exists():
        raise SystemExit('Formal output already exists; refusing implicit resume or overwrite')
    if not website_ok():
        raise SystemExit('Website is unhealthy before launch')
    if shutil.disk_usage(RUN).free < 100 * 1024**3:
        raise SystemExit('Less than 100 GiB free on experiment volume')
    gpu = gpu_inventory()
    if set(gpu) != {0, 1, 2, 3}:
        raise SystemExit('GPU index mapping changed')
    before = gpu_apps(gpu)
    baseline = {}
    for item in before:
        if item['gpu'] not in (1, 2, 3):
            continue
        cmd = ' '.join(item['command'])
        if 'mira_superhuman_demo' not in cmd:
            raise SystemExit(f'Unknown GPU process before launch: {item}')
        baseline[(item['gpu'], item['pid'])] = item['birth']
    if gpu[1]['free_mib'] < 18000 or gpu[2]['free_mib'] < 18000 or gpu[3]['free_mib'] < 18000:
        raise SystemExit('Less than 18 GiB free on a planned training GPU')
    OUT.mkdir(parents=True)
    write_heartbeat()
    environment = os.environ.copy()
    environment.update({
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES': ','.join(gpu[i]['uuid'] for i in (1, 2, 3)),
        'PYTHONPATH': ZARR_DEPS + ':' + DINO,
        'OMP_NUM_THREADS': '2',
        'STAGE1_GUARD_HEARTBEAT': str(HEARTBEAT),
    })
    command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
               str(RUN / 'scripts' / 'train_B_full.py'),
               '--manifest', str(RUN / 'data' / 'pretrain_all_mitoverse.json'),
               '--config', str(RUN / 'configs' / 'B_full.yaml'),
               '--weight', WEIGHT, '--output', str(OUT), '--steps', '100000', '--seed', '0']
    with TRAIN_LOG.open('w') as stream, LOG.open('a') as audit:
        process = subprocess.Popen(command, cwd=OUT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(STATUS, {'stage': 'B_full_running', 'B_full': 'running',
                             'formal_admission': 'passed', 'training_pid': process.pid,
                             'authorized_gpus': [1, 2, 3], 'forbidden_gpu': 0,
                             'manifest_sha256': '3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9',
                             'started_at_unix': time.time(), 'topology_feasibility': 'in_preparation'})
        reason = None
        while process.poll() is None:
            gpu = gpu_inventory()
            apps = gpu_apps(gpu)
            website = website_ok()
            unknown = []
            own_on_gpu0 = []
            for item in apps:
                if is_own_process(item['pid'], process.pid):
                    if item['gpu'] == 0:
                        own_on_gpu0.append(item)
                    continue
                if item['gpu'] in (1, 2, 3) and baseline.get((item['gpu'], item['pid'])) != item['birth']:
                    unknown.append(item)
            missing_website = [(index, pid) for (index, pid), birth in baseline.items()
                               if not any(item['gpu'] == index and item['pid'] == pid and item['birth'] == birth
                                          for item in apps)]
            snapshot = {'time_unix': time.time(), 'training_pid': process.pid, 'gpu': gpu,
                        'apps': apps, 'website_http_200': website, 'unknown_apps': unknown,
                        'website_process_missing': missing_website, 'own_process_on_gpu0': own_on_gpu0,
                        'disk_free_bytes': shutil.disk_usage(RUN).free}
            audit.write(json.dumps(snapshot) + '\n')
            audit.flush()
            write_heartbeat()
            if unknown or missing_website or own_on_gpu0 or not website or snapshot['disk_free_bytes'] < 50 * 1024**3:
                reason = {'unknown_apps': unknown, 'website_process_missing': missing_website,
                          'own_process_on_gpu0': own_on_gpu0, 'website_ok': website,
                          'disk_free_bytes': snapshot['disk_free_bytes']}
                atomic_json(OUT / 'STOP_REASON.json', reason)
                (OUT / 'STOP').write_text('Resource guard requested graceful stop\n')
                break
            time.sleep(60)
        if reason:
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        code = process.wait()
        outcome = {'stage': 'B_full_running' if code == 0 else 'B_full_stopped',
                   'B_full': 'complete' if code == 0 else 'stopped',
                   'exit_code': code, 'stop_reason': reason, 'ended_at_unix': time.time(),
                   'topology_feasibility': 'in_preparation'}
        atomic_json(STATUS, outcome)
        return code


if __name__ == '__main__':
    raise SystemExit(main())
