"""Start guarded frozen probes only after the four-arm SSL service exits complete."""

import json
import subprocess
import time
from pathlib import Path

RUN=Path(__file__).resolve().parent.parent
SRC=RUN/'src'


def service_active(unit):
    return subprocess.run(['systemctl','is-active','--quiet',unit],check=False).returncode==0


def main():
    while True:
        status_path=SRC/'formal_seed0_status.json'
        if status_path.exists():
            state=json.loads(status_path.read_text())['status']
            if state=='blocked':
                raise SystemExit('Formal SSL blocked; downstream dispatch prohibited')
            if state=='completed' and not service_active('mito-pretrain-v2-seed0'):
                if (SRC/'STOP.json').exists():
                    raise SystemExit('STOP marker blocks downstream dispatch')
                probes=SRC/'probes_status.json'
                if probes.exists() and json.loads(probes.read_text())['status']=='completed':
                    return
                if service_active('mito-pretrain-v2-probes'):
                    return
                command=(f'exec /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u '
                         f'{SRC}/coordinator_probes.py >> {SRC}/probes_coordinator.log 2>&1')
                subprocess.run(['systemd-run','--unit=mito-pretrain-v2-probes',
                                f'--property=WorkingDirectory={SRC}',
                                '--property=KillMode=control-group','--property=Restart=no',
                                '/bin/bash','-c',command],check=True)
                time.sleep(5)
                if not service_active('mito-pretrain-v2-probes'):
                    raise SystemExit('Probe service did not become active')
                return
        time.sleep(60)


if __name__=='__main__':
    main()
