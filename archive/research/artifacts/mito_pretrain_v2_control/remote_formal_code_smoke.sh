set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
cd "$run/src"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python coordinator_smoke.py --arm B --gpu 3 --batch 16 --steps 5 --attempt 3
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" <<'PY'
import pathlib, sys, torch
run=pathlib.Path(sys.argv[1]); job=run/'src/jobs/B_seed0_batch16_smoke5_attempt3'
x=torch.load(job/'checkpoint_step5.pth',map_location='cpu',mmap=True,weights_only=False)
assert x['method_hashes'] and sum(x['source_exposure'].values())==80
assert len(x['unique_volumes'])>=1 and x['next_sample']==80
print({'method_hashes':x['method_hashes'],'source_exposure':x['source_exposure'],
       'unique_volumes':len(x['unique_volumes']),'next_sample':x['next_sample']})
PY
