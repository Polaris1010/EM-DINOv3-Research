set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$TASK" <<'PY'
import pathlib,sys,json,ast,torch,importlib
r=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(r/'src'))
for p in [r/'src/validation.py',r/'src/large_hd95.py',r/'scripts/run_validation.py']:ast.parse(p.read_text())
from large_hd95 import percentile,compatibility,MAX_TORCH_QUANTILE_VALUES
module=importlib.import_module('monai.metrics.hausdorff_distance');original=module._compute_percentile_hausdorff_distance
torch.set_num_threads(2)
small=torch.tensor([0.,3.,5.,11.,19.,27.,39.])
with compatibility():assert torch.equal(original(small,95),module._compute_percentile_hausdorff_distance(small,95))
assert module._compute_percentile_hausdorff_distance is original
large=torch.arange(MAX_TORCH_QUANTILE_VALUES+17,dtype=torch.float64)
try:original(large,95)
except RuntimeError as e:assert 'too large' in str(e)
else:raise AssertionError('Expected original size limit for this diagnostic')
actual=float(percentile(large,95,original));expected=(large.numel()-1)*.95
assert abs(actual-expected)<1e-8
evidence={'status':'PASS','small_vector_original_unchanged':True,'binding_restored':True,
 'oversized_vector_values':large.numel(),'original_size_limit_reproduced':True,
 'linear_95th_percentile':actual,'analytic_expected':expected,
 'change_scope':'CPU oversized HD95 percentile only; inference, edges, distances, NSD and MV1 reused metrics unchanged'}
(r/'evidence/hd95_size_limit_fix.json').write_text(json.dumps(evidence,indent=2))
print('HD95_COMPATIBILITY_PASS',evidence)
PY
systemctl reset-failed mv-dev-v2-notta-20261013.service
systemctl start mv-dev-v2-notta-20261013.service
systemctl is-active mv-dev-v2-notta-20261013.service
