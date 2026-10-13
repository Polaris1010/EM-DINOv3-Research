"""Independent verification of the delivered result table and comparison statistics."""
from pathlib import Path
import csv,json,hashlib,math,statistics
R=Path(__file__).resolve().parents[1]
def read(name):return json.loads((R/name).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def table(name):return list(csv.DictReader((R/name).open(encoding='utf8',newline='')))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def main():
 p=read('MV_frozen_probe_protocol.yaml');m=read('MV-Dev-v1.json')
 assert p['status']=='locked_after_P0_pilot' and p['locked_steps']==20000
 dev={v['id']:v for v in m['entries'] if v['supervised_role']=='dev'}
 qc={v['id']:v for v in read('evidence/pixel_QC.json')['volumes']}
 assert len(dev)==4 and len(m['entries'])==8
 models={v['id']:v for v in read('model_registry.json')['models']}
 rows=table('MV_frozen_probe_results.csv');assert len(rows)==36
 expected={(model,str(seed),vid) for model in models for seed in p['seed_sequence'] for vid in dev}
 actual={(r['model'],r['seed'],r['volume_id']) for r in rows}
 assert expected==actual and len(actual)==len(rows)
 for r in rows:
  tp,fp,fn,tn=(int(r[k]) for k in ['tp','fp','fn','tn'])
  assert tp+fp+fn+tn==math.prod(dev[r['volume_id']]['shape_zyx'])
  assert tp+fn==qc[r['volume_id']]['foreground_voxels']
  assert fp+tn==qc[r['volume_id']]['total_voxels']-qc[r['volume_id']]['foreground_voxels']
  close(r['dice'],2*tp/(2*tp+fp+fn));close(r['recall'],tp/(tp+fn))
  if tp+fp:close(r['precision'],tp/(tp+fp))
  assert r['supervised_role']=='dev' and r['ssl_exposed']=='True'
  assert r['acquisition_independence']=='UNVERIFIED' and int(r['locked_steps'])==20000
  assert 0<=float(r['nsd_16nm'])<=1 and float(r['hd95_nm'])>=0
  for axis,s in zip(['z','y','x'],dev[r['volume_id']]['spacing_zyx_nm']):close(r['spacing_'+axis+'_nm'],s)
 metrics=['dice','hd95_nm','nsd_16nm','precision','recall']
 byseed=table('source_metrics_by_seed.csv');triples=table('source_metrics_three_seeds.csv')
 for r in byseed:
  subset=[x for x in rows if all(x[k]==r[k] for k in ['model','seed','source_id'])]
  assert len(subset)==int(r['n_dev_volumes'])
  for k in metrics:
   vals=[float(x[k]) for x in subset if x[k]!=''];close(r[k+'_mean'],statistics.mean(vals))
   if len(vals)>1:close(r[k+'_sd_volumes'],statistics.stdev(vals))
   else:assert r[k+'_sd_volumes']==''
 for r in triples:
  subset=[x for x in byseed if x['model']==r['model'] and x['source_id']==r['source_id']]
  assert len(subset)==3
  for k in metrics:
   vals=[float(x[k+'_mean']) for x in subset];close(r[k+'_mean'],statistics.mean(vals));close(r[k+'_sd_seeds'],statistics.stdev(vals))
 index={(r['model'],r['seed'],r['volume_id']):r for r in rows}
 for d in table('paired_volume_deltas.csv'):
  a=index[d['model'],d['seed'],d['volume_id']];b=index[d['baseline'],d['seed'],d['volume_id']]
  for k in metrics:
   if a[k] and b[k]:close(d['delta_'+k],float(a[k])-float(b[k]))
 run_checks=[]
 for seed in p['seed_sequence']:
  metas=[read(f'runs/{model}/seed{seed}/metrics.json') for model in models]
  for key in ['decoder_initial_sha256','samples_sequence_sha256','probe_source_sha256','manifest_sha256','protocol_sha256']:
   assert len({v[key] for v in metas})==1,key
  for v in metas:
   assert v['status']=='complete' and v['encoder_frozen'] and not v['gates'] and not v['test_labels_accessed']
   assert v['completed_steps']==20000 and v['steps']==20000
   assert v['encoder_sha256']==models[v['model']]['sha256']
   assert v['manifest_sha256']==sha(R/'MV-Dev-v1.json') and v['protocol_sha256']==sha(R/'MV_frozen_probe_protocol.yaml')
   assert v['probe_source_sha256']==sha(R/'src/mv_probe.py')
  run_checks.append({'seed':seed,'head_init_matched':True,'sample_sequence_matched':True,'runtime_code_matched':True})
 assert sha(R/'src/official_primus.py')==next(iter(models.values()))['primus_source_sha256']
 integrity=read('evidence/protected_integrity.json');assert integrity['status']=='PASS' and integrity['before']==integrity['after']
 assert read('evidence/single_seed_result_audit.json')['status']=='PASS'
 assert read('STATUS.json')['status']=='complete' and read('analysis_completion.json')['rows']==36
 receipt={'status':'PASS','rows':36,'independent_confusion_formula_check':True,'source_means_volume_sd_seed_sd_recomputed':True,'paired_deltas_recomputed':True,'run_checks':run_checks,'protected_hashes_unchanged':True,'result_csv_sha256':sha(R/'MV_frozen_probe_results.csv')}
 (R/'evidence/delivery_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt))
if __name__=='__main__':main()
