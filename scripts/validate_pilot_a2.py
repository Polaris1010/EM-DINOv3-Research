"""Reconcile archived A2 tables, paired intervals and unchanged experiment history."""
import csv,json,math,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'experiments/Pilot-A2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def table(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10),(a,b)
def quantile(a,q):
    a=sorted(v for v in a if math.isfinite(v));x=(len(a)-1)*q;k=math.floor(x);return a[k]*(1-x+k)+a[min(k+1,len(a)-1)]*(x-k)
def main():
    manifest=read(R/'evidence/Pilot-A2/delivery_manifest.json')
    for row in manifest['files']:
        rel=row['relative'];p=R/('evidence/Pilot-A2/'+rel[9:] if rel.startswith('evidence/') else 'experiments/Pilot-A2/'+rel)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
    assert read(R/'evidence/Pilot-A2/inputs_before.json')==read(R/'evidence/Pilot-A2/inputs_after.json')
    assert (A/'tables/baseline_pixel_discrimination.csv').read_bytes()==(A/'tables/final_pixel_discrimination.csv').read_bytes()
    b=read(A/'train_only_bias.json');samples=read(R/'experiments/Feature-Pilot-A/samples.json');train=[x for x in samples if x['role']=='train']
    assert b['train_sample_ids']==[x['id'] for x in train] and len(train)==312
    assert set(b['train_regions'])=={x['region_id'] for x in train} and len(b['train_regions'])==156
    assert not b['evaluation_GT_read'] and not b['baseline_metrics_read']
    prior=sum(x['foreground_pixels']/x['valid_pixels'] for x in b['train_sources'].values())/12
    for model,x in b['models'].items():
        close(x['source_equal_foreground_prior'],prior);close(x['threshold'],1/(1+math.exp(x['bias'])))
        assert abs(x['gradient_residual'])<1e-11 and x['weighted_BCE_fit']<x['weighted_BCE_raw']
    pixels=table(A/'tables/final_pixel_metrics.csv');pooled=table(A/'tables/final_source_metrics.csv');assert len(pixels)==1280 and len(pooled)==20
    old=table(R/'experiments/Feature-Pilot-A/tables/source_readout_metrics.csv')
    keys=['tp','fp','fn','valid_pixels','foreground_pixels','predicted_pixels','far_background_pixels','far_FP_pixels','brier_sum']
    metric=lambda d,k: {'dice':2*d['tp']/(2*d['tp']+d['fp']+d['fn']),'precision':d['tp']/(d['tp']+d['fp']),'recall':d['tp']/(d['tp']+d['fn']),'far_FP_rate':d['far_FP_pixels']/d['far_background_pixels'],'brier':d['brier_sum']/d['valid_pixels']}[k]
    for row in pooled:
        subset=[x for x in pixels if all(x[k]==row[k] for k in ['model','source','operation'])];assert len(subset)==64
        d={k:sum(float(x[k]) for x in subset) for k in keys}
        for k in keys:close(row[k],d[k])
        assert d['predicted_pixels']==d['tp']+d['fp'] and d['foreground_pixels']==d['tp']+d['fn']
        for k in ['dice','precision','recall','far_FP_rate','brier']:close(row[k],metric(d,k))
        if row['operation']=='raw':
            prev=next(x for x in old if x['model']==row['model'] and x['source']==row['source'] and x['layer']=='23')
            for k in ['tp','fp','fn','valid_pixels','far_background_pixels','far_FP_pixels']:close(row[k],prev[k])
    def m(source,model,op,key):return float(next(x[key] for x in pooled if (x['source'],x['model'],x['operation'])==(source,model,op)))
    rank=table(A/'tables/final_pixel_discrimination.csv');contrasts=table(A/'tables/paired_changes.csv')
    for source in sorted({x['source'] for x in pooled}):
        draws=table(A/'tables'/('bootstrap_draws_'+source+'.csv'));assert len(draws)==200
        for row in [x for x in contrasts if x['source']==source]:
            name=row['contrast'];values=[float(x[name]) for x in draws]
            close(row['bootstrap_low'],quantile(values,.025));close(row['bootstrap_high'],quantile(values,.975))
            if name.endswith(('_AUROC_delta','_AP_delta')):
                group,key=name.rsplit('_',2)[0],name.rsplit('_',2)[1]
                def rankval(model):return float(next(x[key] for x in rank if x['source']==source and x['scope']==group and x['model']==model))
                expected=rankval('D2-100k')-rankval('D2-30k')
            elif name=='Dice_gap_recovery_R':
                expected=(m(source,'D2-100k','calibrated','dice')-m(source,'D2-30k','calibrated','dice'))-(m(source,'D2-100k','raw','dice')-m(source,'D2-30k','raw','dice'))
            elif name.startswith('100k_cal_minus_raw_'):
                key=name[len('100k_cal_minus_raw_'):];expected=m(source,'D2-100k','calibrated',key)-m(source,'D2-100k','raw',key)
            else:
                text=name[len('100k_minus_30k_'):];op,key=text.split('_',1);expected=m(source,'D2-100k',op,key)-m(source,'D2-30k',op,key)
            close(row['estimate'],expected)
    registry=read(R/'experiments/registry.yaml');base=json.loads(subprocess.check_output(['git','-c','safe.directory='+R.as_posix(),'show','2a2a347f8630c30295a1d3961ef6c8ff5be187b2:experiments/registry.yaml'],cwd=R))
    new={x['experiment_id']:x for x in registry['experiments']}
    for x in base['experiments']:assert new[x['experiment_id']]==x,x['experiment_id']
    assert set(new)-{x['experiment_id'] for x in base['experiments']}=={'Pilot-A2'}
    out={'status':'PASS','downloaded_SHA_verified':len(manifest['files']),'historical_inputs_unchanged':True,'baseline_counts_match_Pilot_A':True,'pixel_rows_reconciled':1280,'pooled_rows_reconciled':20,'ranking_rows':len(rank),'paired_contrasts_and_intervals':len(contrasts),'historical_registry_entries_unchanged':60,'fit_source_roles':'312 train FOVs /156 regions; no eval labels','GPU_smoke':'evidence/Pilot-A2/numerical_smoke.json'}
    (R/'evidence/Pilot-A2/independent_validation.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
if __name__=='__main__':main()
