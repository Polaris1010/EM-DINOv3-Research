set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
wheel="$run/src/gudhi-3.13.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
sha256sum "$wheel" > "$run/evidence/gudhi_wheel_sha256.txt"
"$base/envs/meddinov3/bin/python" -m pip install --no-deps --no-index --target "$run/deps" "$wheel"
PYTHONPATH="$run/deps" "$base/envs/meddinov3/bin/python" - <<'PY'
import gudhi,numpy as np
print('gudhi_version',gudhi.__version__)
cc=gudhi.CubicalComplex(top_dimensional_cells=np.array([[0.0,0.5],[0.5,1.0]]))
print('ph',cc.persistence())
print('bottleneck_empty',gudhi.bottleneck_distance(np.empty((0,2)),np.empty((0,2))))
PY
