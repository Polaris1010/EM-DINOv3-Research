from pathlib import Path
import json,zipfile,time
BASE=Path(__file__).resolve().parent
names=['locked_inputs.json','repository_state.json','gram_config_audit.json','launch_status.json','process_registry.json','allowed_gpu_processes.json','final_code_sha256.json','resource_log.csv','coordinator_events.jsonl','修订执行方案.md','official_to_local_config_diff.md','ssl_exclusion_report.csv']
with zipfile.ZipFile(BASE/'launch_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in names:
  p=BASE/name
  if p.exists():z.write(p,name)
 for directory in (BASE/'jobs').glob('v4_b16_*'):
  for name in ['result.json','progress.json','identity.json','guard_result.json','resolved_config.yaml','source_sha256.json','steps.jsonl','fixed_feature_diagnostics.jsonl','checkpoint_index.jsonl']:
   p=directory/name
   if p.exists():z.write(p,p.relative_to(BASE))
print('EVIDENCE',BASE/'launch_evidence.zip')
