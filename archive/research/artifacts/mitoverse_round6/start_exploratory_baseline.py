"""User-requested corrected supervised baseline; never unlocks SSL gates."""
import os, json, time, hashlib, shutil, psutil
from settings import *
from guard import Guard, save

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def state(status, **kwargs):
    p = BASE / 'stage_status.json'
    s = json.loads(p.read_text())
    s['exploratory_masked_baseline'] = dict(status=status, time=time.time(), **kwargs)
    save(p, s)

def main():
    prior = json.loads((BASE / 'coordinator_pid.json').read_text())
    if psutil.pid_exists(prior['pid']):
        p = psutil.Process(prior['pid'])
        assert abs(p.create_time() - prior['birth']) > .1 or p.status() == psutil.STATUS_ZOMBIE, 'Coordinator still active'
    assert not (BASE / 'exploratory_baseline_protocol.json').exists(), 'Inspect existing run before recovery'
    assert not (BASE / 'STOP.json').exists(), 'Resolve recorded stop before launch'
    assert json.loads((BASE / 'stage_status.json').read_text())['mask_correction']['status'] == 'completed'
    save(BASE / 'coordinator_before_exploratory.json', prior)
    save(BASE / 'coordinator_pid.json', dict(pid=os.getpid(), birth=psutil.Process().create_time(), command=['start_exploratory_baseline.py'], started=time.time()))
    g = Guard()
    g.start()
    try:
        assert sha(P1) == P1_SHA
        audit = json.loads((BASE / 'masked_schedule_audit.json').read_text())
        for row in audit['seeds']:
            assert sha(BASE / f"train_patch_schedule_masked_seed{row['seed']}.jsonl") == row['sha256']
        split = json.loads((R4 / 'split_mitoverse_probe_v1.json').read_text())
        for vid in split['validation']:
            assert not (Path(split['volumes'][vid]['image_path']).parent / 'mask').exists(), 'Validation evaluator must handle new region masks'
        files = [BASE / 'exploratory_downstream.py', BASE / 'masked_data.py', BASE / 'mask_policy_v2.json', BASE / 'guard.py', BASE / 'worker.py', BASE / 'start_exploratory_baseline.py', R4 / 'common.py', R4 / 'train.py', R4 / 'split_mitoverse_probe_v1.json']
        archive = BASE / 'exploratory_source_snapshot'; archive.mkdir()
        hashes = {}
        for i, p in enumerate(files):
            shutil.copyfile(p, archive / f'{i}_{p.name}'); hashes[str(p)] = sha(p)
        protocol = dict(status='locked_exploratory_only', authorization='User explicitly requested next training after provenance block was reported; corrected supervised P1 baseline only', formal_S0_pass=False, added_SSL=False, independent_benchmark_claim=False, seeds=[0,1,2], steps_per_seed=10000, batch=8, crop=256, norm='N2', encoder='P1 frozen', encoder_sha256=P1_SHA, head_lr=1e-4, betas=[.9,.98], weight_decay=.05, clip_grad=12, schedule='R5 50-step warmup; 10k cosine floor .1', validation='fixed 8 evenly spaced slices per volume every 1000 steps; full validation after each finished head; legacy test sealed for this branch', schedules=audit, source_hashes=hashes, head_hashes={str(s):sha(R4/f'shared_head_init_seed{s}.pth') for s in range(3)}, execution='one GPU2 task at a time; serial I/O; unified existing guard')
        save(BASE / 'exploratory_baseline_protocol.json', protocol)
        state('running', phase='independent_batch8_smoke', added_SSL=False)
        g.job('masked_P1_batch8_smoke10', 2, 'exploratory_downstream.py', ['--seed','0','--steps','10'])
        for seed in range(3):
            state('running', seed=seed, phase='train', steps_per_seed=10000, added_SSL=False)
            out = g.job(f'masked_P1_seed{seed}_10000', 2, 'exploratory_downstream.py', ['--seed',str(seed),'--steps','10000'])
            state('running', seed=seed, phase='full_validation', added_SSL=False)
            g.job(f'masked_P1_seed{seed}_validation', 2, 'exploratory_downstream.py', ['--seed',str(seed),'--evaluate-from',str(out),'--validation-only'])
        state('completed', seeds=[0,1,2], steps_per_seed=10000, added_SSL=False, interpretation='Corrected supervised exploratory baseline; no SSL effect or independent benchmark claim')
    except Exception as e:
        g.stop_all(repr(e)); state('blocked', reason=repr(e), added_SSL=False); raise
    finally:
        g.close()

if __name__ == '__main__': main()
