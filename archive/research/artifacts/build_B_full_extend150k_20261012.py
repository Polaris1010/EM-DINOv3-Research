"""Build a separate, explicit 100k->150k continuation from audited Stage1 code."""
from pathlib import Path
import json,hashlib,shutil,ast
OLD=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
STAGE2A2=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
NEW=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
PARENT=OLD/'runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth'
PARENT_SHA='12bbf09ffcd7f93cbad88c4e55df55d36b3a299990ad976bf29af51a118a265b'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)
def write(p,s):
    compile(s,str(p),'exec');p.write_text(s)

def main():
    assert not NEW.exists(),'Continuation directory already exists'
    assert sha(OLD/'scripts/train_B_full.py')=='dfdd9db6c5bbfc2456e3eba19224cdf3eea305823743caadb00eeec309c2244c'
    for folder in ['scripts','src','configs','data','audit','references','runs/B_full','evaluation']:(NEW/folder).mkdir(parents=True,exist_ok=True)
    for a,b in [(OLD/'scripts/ssl_full_data.py',NEW/'scripts/ssl_full_data.py'),(OLD/'configs/B_full.yaml',NEW/'configs/B_full.yaml'),(OLD/'data/pretrain_all_mitoverse.json',NEW/'data/pretrain_all_mitoverse.json'),(OLD/'src/dinov3_commit.txt',NEW/'src/dinov3_commit.txt'),(STAGE2A2/'references/paths.json',NEW/'references/paths.json'),(STAGE2A2/'src/frozen_probe.py',NEW/'src/frozen_probe.py'),(STAGE2A2/'src/meddinov3_compute_metrics_upstream_20261012.py',NEW/'src/repository_metrics.py')]:shutil.copy2(a,b)
    s=(OLD/'scripts/train_B_full.py').read_text()
    s=replace(s,"MILESTONES = {1000, 5000, 10000, 25000, 50000, 75000, 100000}","MILESTONES = {100010, 105000, 125000, 150000}")
    s=replace(s,'if steps > 100000 or batch != cfg.train.batch_size_per_gpu or world_size not in (1, 3):','if not 100000 < steps <= 150000 or batch != cfg.train.batch_size_per_gpu or world_size != 3:')
    s=replace(s,"'effective_peak_lr': float(cfg.optim.lr),\n                'warmup_steps': 5000, 'total_steps': 100000,","'effective_peak_lr': float(cfg.optim.min_lr),\n                'warmup_steps': 0, 'total_steps': 150000,\n                'schedule': 'constant_original_min_lr_after_100k',\n                'parent_checkpoint_sha256': '"+PARENT_SHA+"',")
    s=replace(s,"        if saved['identity'] != identity:\n            raise ValueError('Cross-run resume identity mismatch')",'''        parent = saved['identity']
        fields = ('manifest_sha256', 'config_sha256', 'official_weight_sha256',
                  'sampler_sha256', 'dinov3_commit', 'world_size', 'batch_per_gpu',
                  'gradient_accumulation', 'effective_global_batch', 'weight_decay',
                  'teacher_momentum', 'seed')
        if any(parent[key] != identity[key] for key in fields):
            raise ValueError('Continuation data/model/optimizer identity mismatch')
        if int(saved['step']) == 100000:
            if sha256(args.resume) != identity['parent_checkpoint_sha256']:
                raise ValueError('Not the authorized B_full 100k checkpoint')
        elif parent != identity:
            raise ValueError('Continuation restart identity mismatch')''')
    s=replace(s,"            **identity, 'initialization': 'official_natural_vitl16', 'strict_backbone_load': True,","            **identity, 'initialization': 'full_state_B_full_100k_continuation', 'strict_backbone_load': True,\n            'restored': ['student','teacher','optimizer','dino_loss','ibot_loss','rng_by_rank','sample_chain'],")
    begin=s.index('            warmup = min(1.0, (step + 1) / 5000)');end=s.index('            for group in optimizer.param_groups:',begin)
    s=s[:begin]+"            lr = float(cfg.optim.min_lr)  # Explicit tail; no cosine reheating.\n"+s[end:]
    s=replace(s,"    start = 0\n    prior_rng = None","    if not args.resume:\n        raise ValueError('Continuation requires a complete checkpoint')\n    start = 0\n    prior_rng = None")
    # Verify restoration itself before the first optimizer update.
    s=replace(s,"        prior_rng = saved['rng_by_rank'][rank]",'''        if len(optimizer.state) != len(saved['optimizer']['state']):
            raise ValueError('Optimizer state was not restored')
        prior_rng = saved['rng_by_rank'][rank]''')
    write(NEW/'scripts/train_B_full.py',s)
    g=(OLD/'scripts/gpu_guard.py').read_text()
    begin=g.index("    admission = json.loads(");end=g.index('    if OUT.exists():',begin)
    replacement="    admission = json.loads((RUN / 'audit/continuation_admission.json').read_text())\n    if admission['status'] != 'admitted':\n        raise SystemExit('Continuation smoke admission missing')\n    for path, key in [(RUN / 'scripts/train_B_full.py','trainer_sha256'), (RUN / 'scripts/ssl_full_data.py','sampler_sha256'), (RUN / 'configs/B_full.yaml','config_sha256'), (RUN / 'data/pretrain_all_mitoverse.json','manifest_sha256')]:\n        if digest(path) != admission[key]:\n            raise SystemExit('Continuation inputs changed after smoke')\n"
    g=g[:begin]+replacement+g[end:]
    g=replace(g,"--steps', '100000', '--seed', '0']","--steps', '150000', '--seed', '0', '--resume', '"+str(PARENT)+"']")
    g=replace(g,"        atomic_json(STATUS, outcome)\n        return code",'''        atomic_json(STATUS, outcome)
        if code == 0:
            atomic_json(STATUS, {**outcome, 'evaluation': 'running'})
            evaluation = subprocess.run([sys.executable, str(RUN / 'scripts/posttrain_evaluation.py')], cwd=RUN)
            atomic_json(STATUS, {**outcome, 'evaluation': 'complete' if evaluation.returncode == 0 else 'failed', 'evaluation_exit_code': evaluation.returncode})
            return evaluation.returncode
        return code''')
    write(NEW/'scripts/gpu_guard.py',g)
    # Freeze-evaluation training loop remains the original matched control loop.
    f=(STAGE2A2/'src/frozen_probe_control.py').read_text()
    f=replace(f,"ENCODER=Path(REF['B_full_teacher_backbone'])","ENCODER=Path(os.environ['B_FULL_CHECKPOINT'])\nENCODER_STEP=int(os.environ.get('B_FULL_STEP','150000'))")
    f=f.replace("ck['step']==100000","ck['step']==ENCODER_STEP")
    f=f.replace("'encoder_step':100000","'encoder_step':ENCODER_STEP")
    f=f.replace("'continual_em_steps':100000","'continual_em_steps':ENCODER_STEP")
    f=f.replace('frozen B_full official DINOv3 ViT-L/16 teacher','frozen B_full official DINOv3 ViT-L/16 teacher')
    write(NEW/'src/frozen_probe_150k.py',f)
    plan={'experiment':'B_full_100k_to_150k','authorized_by_user':True,'parent_checkpoint':str(PARENT),'parent_checkpoint_sha256':PARENT_SHA,'start_step':100000,'target_step':150000,'additional_steps':50000,'schedule':'constant original minimum base LR = 1e-6 after completed 100k cosine','schedule_interpretation':'continuation at original LR floor; not a from-step-0 150k cosine experiment, and not evidence of optimal convergence if no gain','losses':'DINO 1, iBOT 1, KoLeo 0.1','architecture':'official DINOv3 ViT-L/16','world_size':3,'batch_per_gpu':16,'effective_global_batch':48,'data':'same full 212-volume 13-source 48722-slice image-only manifest','teacher_momentum':0.999,'weight_decay':0.04,'preserved_checkpoints':[100000,125000,150000],'evaluation':'automatic 150k frozen CMCC/CMCT, identical decoder protocol and seed 20260907; then MedDINOv3 default Dice/HD95/NSD mean/std; compare original 100k','GPUs':[1,2,3],'forbidden_GPU':0,'P':'paused','new_topology_or_masking_modules':False,'estimated_training_hours_from_prior_throughput':11.125}
    (NEW/'continuation_plan.json').write_text(json.dumps(plan,indent=2))
    (NEW/'STATUS.json').write_text(json.dumps({'B_full':'preparing_resume_smoke','user_authorized_steps':150000,'P':'paused','methods_long_training':'prohibited'},indent=2))
    print('BUILT',str(NEW),flush=True)

if __name__=='__main__':main()
