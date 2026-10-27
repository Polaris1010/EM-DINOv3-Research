"""CPU reconciliation of final C-1 delivery; no model/image loading."""
import json,csv,subprocess,statistics,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'experiments/Pilot-C';E=ROOT/'evidence/Pilot-C'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def table(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
 j=read(R/'PilotC1_summary.json');rows=table(R/'PilotC1_per_rank_steps.csv');steps=table(R/'PilotC1_per_step.csv');pairs=table(R/'PilotC1_pairing_checks.csv')
 assert j['C0_updates']==j['C1_updates']==20 and len(rows)==120 and len(steps)==40 and len(pairs)==60
 for branch in ['C0','C1']:
  for rank in range(3):assert [int(x['step']) for x in rows if x['branch']==branch and int(x['rank'])==rank]==list(range(30001,30021))
 for rank in range(3):
  a=[x for x in rows if x['branch']=='C0' and int(x['rank'])==rank];b=[x for x in rows if x['branch']=='C1' and int(x['rank'])==rank]
  for x,y in zip(a,b):
   assert all(x[k]==y[k] for k in ['tensor_hashes','sample_chain','sample_digest','before_model_RNG','after_model_RNG','lr','weight_decay','batch_first_index'])
 for branch in ['C0','C1']:
  a=[float(x['production_estimate_bottleneck_seconds']) for x in steps if x['branch']==branch and int(x['step'])>=30006]
  assert math.isclose(statistics.median(a),j['throughput'][branch]['production_estimate_bottleneck_seconds'],rel_tol=1e-12)
 b=[x for x in rows if x['branch']=='C1'];assert all(float(x['barrier_raw'])>0 and float(x['barrier_backbone_grad_norm_local'])>0 and x['anchor_gradients_present']=='False' and float(x['diagnostic_student_tokens_max_difference'])==0 for x in b)
 assert math.isclose(statistics.median(float(x['active_fraction']) for x in b),j['barrier']['active_fraction']['median'],rel_tol=1e-12)
 proof=read(E/'paired_consistency.json');stop=read(E/'stopped_receipt.json');protocol=read(ROOT/'configs/pretrain/Pilot_C1_protocol.json')
 assert proof['historical_C0_loss_and_sampling_exact'] and proof['coordinate_max_abs_difference']==0 and stop['exit_code']==0 and not stop['service_active']
 assert not protocol['formal_15000_SSL_authorized'] and not protocol['Primus_authorized'] and protocol['barrier']['smoke_lambda']==1
 assert not j['formal_SSL_or_Primus_started'] and not j['further_experiments_started'] and not stop['status']['GPU0_used']
 source=read(E/'runtime_source.json')
 for name,digest in source['source_sha256'].items():assert hashlib.sha256((ROOT/'src/pilot_c1'/name).read_bytes()).hexdigest()==digest,name
 previous=json.loads(subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show','ba6b4d967600734ee0a5e38fc91157c717330213:experiments/registry.yaml'],cwd=ROOT));new=read(ROOT/'experiments/registry.yaml');idx={x['experiment_id']:x for x in new['experiments']}
 for x in previous['experiments']:assert idx[x['experiment_id']]==x,x['experiment_id']
 assert set(idx)-{x['experiment_id'] for x in previous['experiments']}=={'Pilot-C1'}
 result={'status':'PASS','rank_step_rows':120,'paired_rank_steps':60,'per_step_rows':40,'prior_registry_records_unchanged':62,'runtime_source_SHA_match':True,'max_updates_per_branch':20,'formal_training_started':False}
 (E/'independent_validation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
if __name__=='__main__':main()
