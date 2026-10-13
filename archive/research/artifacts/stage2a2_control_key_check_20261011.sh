set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import csv,json
from pathlib import Path
root=Path.cwd();out=root/'branch_S/blind_review_raw_20261011'
responses=list(csv.DictReader((out/'human_review_P001_P020_overlay_20261011.csv').open()))
ids={r['pair_id'] for r in responses}
keys=[r for r in csv.DictReader((root/'branch_S/blind_review/answer_key_CONCEALED.csv').open()) if r['pair_id'] in ids]
scores={(int(r['sample_index']),int(r['candidate'])):float(r['cut_mean']) for r in csv.DictReader((root/'branch_S/graph_cut/same_shape_scores_424.csv').open())}
for k in keys:
    a=scores[int(k['sample_index']),int(k['A_candidate'])];b=scores[int(k['sample_index']),int(k['B_candidate'])]
    assert a==float(k['score_A']) and b==float(k['score_B'])
    assert k['score_winner']==('A' if a<b else 'B')
check={'status':'pass','pairs':len(keys),'key_matches_current_cut_mean':True,'lower_affinity_is_prelocked_winner':True}
(out/'human_key_check.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check))
PY
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/microsam/bin/python src/fresh_trial20.py
