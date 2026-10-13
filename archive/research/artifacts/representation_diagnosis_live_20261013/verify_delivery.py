"""Validate delivered live evidence, without CUDA, training or remote writes."""
from pathlib import Path
import csv,json,hashlib,re
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    live=load('evidence/live_evidence.json')
    smoke=load('smoke/real_weight_smoke.json')
    immutable=load('smoke/immutable_inputs_check.json')
    params=load('smoke/model_parameter_summary.json')
    inv=load('evidence/checkpoint_inventory_verified.json')
    state=load('evidence/final_server_state.json')
    plan=load('evidence/diagnosis_plan.unlocked.json')
    assert smoke['status']=='PASS' and smoke['physical_gpu']==3 and not smoke['long_training_started']
    assert immutable['status']=='PASS' and immutable['before']==immutable['after']
    assert immutable['checkpoint_hashes_before']==immutable['checkpoint_hashes_after']
    for source in live['sources']:
        if 'sha256' in source:
            p=R/'sources'/Path(source['copy']).name
            assert sha(p)==source['sha256']
    for source in state['copied_sources']:
        assert sha(R/'evidence'/Path(source['copy']).name)==source['sha256']
    assert {m['model'] for m in smoke['models']}=={'P0','B100k','B150k'}
    assert len(live['historical_probes'])==6 and all(m['status']=='complete' for m in live['historical_probes'])
    for m in live['historical_probes']:
        assert sha(R/'evidence'/f"{m['id']}_{m['dataset']}_metrics.json")==m['sha256']
        assert m['steps']==3000 and m['seed']==20260907
    assert state['website_http']==200
    assert [i['step'] for i in inv]==[1000,5000,10000,25000,50000,75000,100000,100010,105000,125000,150000]
    for spec in plan['models']:
        assert spec['sha256']==immutable['checkpoint_hashes_before'][spec['id']]
    assert plan['status']=='UNVERIFIED_ROIS_NOT_LOCKED' and plan['rois']==[]
    for m in params:
        assert m['encoder_parameters']==303154176 and m['encoder_trainable_parameters']==0
        assert m['head_parameters']==46691082 and m['frequency_gate_parameters']==2114568
    paired={}
    for m in smoke['models']:
        assert m['encoder_unchanged'] and len(m['resolution_views'])==6
        assert all(r['optimizer_steps']==2 and not r['encoder_gradients'] for r in m['readout_training'].values())
        for v in m['resolution_views']:
            assert not v['segmentation_metric_computed']
            paired.setdefault((v['mode'],v['input_size']),set()).add(v['input_tensor_sha256'])
    assert all(len(v)==1 for v in paired.values())
    for md in R.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',md.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:')):continue
            assert (md.parent/target).exists(),(md.name,target)
    result={'status':'PASS','server_real_weight_smoke':'PASS','head_optimizer_updates':12,'real_em_resolution_forwards':18,
        'verified_encoder_exports':len(inv),'formal_ssl_manifest_sha256':sha(R/'evidence/formal_pretrain_manifest.json'),
        'immutable_inputs':'PASS','website_http_after':200,'physical_gpu':3,
        'full_multiseed_benchmark_completed':False,'physical_segmentation_resolution_benchmark_completed':False,
        'remaining_unverified':['annotation completeness/valid regions','acquisition isolation','CMCC/CMCT physical calibration']}
    (R/'verification_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    metrics=load('evidence/final_results.json')['summary']
    with (R/'historical_label1_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    summaries=[{k:v for k,v in m.items() if k!='trainable_named_parameters'} for m in params]
    (R/'model_parameter_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    files=[p for p in R.rglob('*') if p.is_file() and p.name!='delivery_sha256.json' and '__pycache__' not in p.parts]
    (R/'delivery_sha256.json').write_text(json.dumps([{'file':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)],indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
