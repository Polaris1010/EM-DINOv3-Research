"""Fixed45k parent identity and deterministic continuation boundary."""
MODELS=['C0_Vanilla_100k','C1_Barrier_100k']
ANCHOR_EMA='7db12e7ff8bee45166aaeeda1d94365e696ddaecf0c57a0d1611a891cf0d3eda'
def validate_protocol(p):
 assert p['models']==MODELS
 assert (p['initial_step'],p['final_step'],p['additional_updates_per_model'])==(45000,100000,55000)
 assert (p['lambda'],p['margin'],p['reference_quantile'],p['Chebyshev_strictly_greater_than'],p['layer'])==(1,.05,.5,2,23)
 assert not p['diagnostic_VJP'] and not p['diagnostic_extra_student_forward']
 assert p['parent_EMA_sha256']==ANCHOR_EMA
 assert p['anchor_EMA_relative']=='codex/ssl_diversity_d12_20261017/runs/D2/formal/checkpoints/step000030000/teacher_backbone.pth'
 assert p['world_size']==3 and p['global_batch']==48 and p['Primus_steps']==40000 and p['Primus_seed']==20261014 and not p['TTA']
def validate_parent(saved,model,p):
 validate_protocol(p);assert model in MODELS
 identity=saved['identity'];parent=p['parents'][model]
 assert saved['step']==45000 and identity['stage']=='Pilot-C2' and identity['model']==parent['model_45k']
 assert identity['initial_step']==30000 and identity['final_step']==45000
 assert identity['barrier_enabled']==(model=='C1_Barrier_100k')
 assert identity['protocol_sha256']==p['parent_45k_protocol_sha256'] and identity['source_commit']==p['parent_45k_source_commit']
 assert identity['parent_full_sha256']==p['parent_full_sha256']
 assert identity['global_batch']==48 and identity['world_size']==3 and identity['legacy_config_sha256']==p['legacy_SSL_config_sha256']
 assert len(saved['rng_by_rank'])==3 and saved['sample_chain']==p['initial_sample_chain']
 assert sorted({int(v['step']) for v in saved['optimizer']['state'].values()})==[45000]
 for key in ['student','teacher','dino_loss','ibot_loss']:assert key in saved
def validate_boundary(previous,chain,rng,steps):
 assert previous['step']==45000 and previous['sample_chain']==chain
 assert previous['after_model_RNG']==rng and steps==[45000]