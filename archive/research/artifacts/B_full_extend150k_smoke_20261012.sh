set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012
cd "$RUN"
test ! -e runs/B_full/resume_smoke10
env OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u - <<'PY'
from pathlib import Path
import os,json,hashlib,time,subprocess,signal,importlib.util,math
root=Path.cwd();spec=importlib.util.spec_from_file_location('guard',root/'scripts/gpu_guard.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
assert g.website_ok()
inv=g.gpu_inventory();apps=g.gpu_apps(inv);baseline={(a['gpu'],a['pid']):a['birth'] for a in apps if a['gpu'] in (1,2,3)}
for a in apps:
    if a['gpu'] in (1,2,3):assert 'mira_superhuman_demo' in ' '.join(a['command'])
assert all(inv[i]['free_mib']>=18000 for i in (1,2,3))
env=os.environ.copy();env.update({'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':','.join(inv[i]['uuid'] for i in (1,2,3)),'PYTHONPATH':g.ZARR_DEPS+':'+g.DINO,'OMP_NUM_THREADS':'2','STAGE1_GUARD_HEARTBEAT':str(root/'audit/smoke_heartbeat.txt')})
parent=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth')
cmd=[g.PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',str(root/'scripts/train_B_full.py'),'--manifest',str(root/'data/pretrain_all_mitoverse.json'),'--config',str(root/'configs/B_full.yaml'),'--weight',g.WEIGHT,'--output',str(root/'runs/B_full/resume_smoke10'),'--steps','100010','--seed','0','--resume',str(parent)]
heartbeat=Path(env['STAGE1_GUARD_HEARTBEAT']);heartbeat.write_text(str(time.time()))
with (root/'audit/resume_smoke10.log').open('w') as f:
    p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print('SMOKE_PID',p.pid,flush=True);last=''
    while p.poll() is None:
        heartbeat.write_text(str(time.time()))
        current=g.gpu_apps(g.gpu_inventory())
        bad=[a for a in current if (g.is_own_process(a['pid'],p.pid) and a['gpu']==0) or (not g.is_own_process(a['pid'],p.pid) and a['gpu'] in (1,2,3) and baseline.get((a['gpu'],a['pid']))!=a['birth'])]
        if bad or not g.website_ok():
            os.killpg(p.pid,signal.SIGTERM);p.wait();raise RuntimeError('Smoke GPU/website guard failed')
        progress=root/'runs/B_full/resume_smoke10/progress.json'
        if progress.exists():
            s=progress.read_text()
            if s!=last:print('SMOKE_PROGRESS',s,flush=True);last=s
        time.sleep(5)
    if p.returncode:
        print((root/'audit/resume_smoke10.log').read_text()[-12000:],flush=True)
        raise SystemExit(p.returncode)
out=root/'runs/B_full/resume_smoke10';r=json.loads((out/'result.json').read_text());init=json.loads((out/'initialization.json').read_text())
assert r['status']=='complete' and r['completed_steps']==100010 and r['max_rank_parameter_difference']==0
assert r['tracked_student_delta_l2']>0 and r['tracked_teacher_delta_l2']>0
assert init['starting_step']==100000 and init['effective_global_batch']==48 and init['schedule']=='constant_original_min_lr_after_100k'
for rank in range(3):
    rows=[json.loads(s) for s in (out/f'steps_rank{rank}.jsonl').read_text().splitlines()]
    assert len(rows)==10 and [x['step'] for x in rows]==list(range(100001,100011))
    assert all(x['data_first_index']==(x['step']-1)*48+rank*16 and x['lr']==1e-6 and math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ad={'status':'admitted','parent_step':100000,'smoke_completed_step':100010,'smoke_world_size':3,'sampler_continuation_indices':'pass','restored_full_state':'pass','finite_loss_and_gradient':'pass','inter_rank_difference':0,'trainer_sha256':sha(root/'scripts/train_B_full.py'),'sampler_sha256':sha(root/'scripts/ssl_full_data.py'),'config_sha256':sha(root/'configs/B_full.yaml'),'manifest_sha256':sha(root/'data/pretrain_all_mitoverse.json'),'resume_sha256':'12bbf09ffcd7f93cbad88c4e55df55d36b3a299990ad976bf29af51a118a265b','lr_policy':'constant min LR 1e-6 tail','website_http_200':g.website_ok()}
(root/'audit/continuation_admission.json').write_text(json.dumps(ad,indent=2))
print('CONTINUATION_ADMITTED',json.dumps(ad),flush=True)
PY
