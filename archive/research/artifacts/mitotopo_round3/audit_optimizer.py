"""Validate learning rates and weight decay in saved optimizer state, not just YAML."""
from pathlib import Path
import json,math,torch,gc
BASE=Path(__file__).resolve().parent;torch.set_num_threads(2);rows={}
for p in sorted((BASE/'jobs').glob('ssl_*')):
 if not (p/'result.json').exists() or not (p/'checkpoint_latest.pth').exists():continue
 r=json.loads((p/'result.json').read_text())
 if r['status']!='complete':continue
 ck=torch.load(p/'checkpoint_latest.pth',map_location='cpu',mmap=True,weights_only=False);cfg=ck['config'];groups=ck['optimizer']['param_groups'];bs=r['batch_size'];base=r['base_lr']
 assert ck['completed_steps']==r['completed_steps'];assert cfg['train']['batch_size_per_gpu']==bs
 for g in groups:
  assert math.isclose(g['lr'],base*g['lr_multiplier'],rel_tol=1e-10)
  assert math.isclose(g['weight_decay'],.04*g['wd_multiplier'],rel_tol=1e-10,abs_tol=1e-15)
 official=float(cfg['schedules']['lr']['end'])*4*math.sqrt(bs/1024)
 rows[p.name]={'completed_steps':r['completed_steps'],'batch_size':bs,'effective_gram_end':cfg['gram']['loss_weight_schedule']['end'],'saved_optimizer_base_lr':base,'official_scaled_base_lr_from_nominal_yaml':official,'actual_to_official_ratio':base/official,'intentional_lr_factor_override':r.get('lr_factor',4),'parameter_groups_checked':len(groups),'group_lr_min':min(g['lr'] for g in groups),'group_lr_max':max(g['lr'] for g in groups),'lr_and_weight_decay_checks':'pass'}
 del ck;gc.collect()
(BASE/'effective_optimizer_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
