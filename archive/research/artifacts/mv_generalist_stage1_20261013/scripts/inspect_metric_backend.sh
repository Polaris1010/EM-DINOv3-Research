set -e
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import inspect,shutil
from monai.metrics.utils import get_surface_distance,get_edge_surface_distance
from monai.transforms.utils import distance_transform_edt
print('PROFILERS',shutil.which('py-spy'),shutil.which('gdb'),shutil.which('perf'))
print(inspect.getsource(get_surface_distance))
print(inspect.getsource(get_edge_surface_distance))
print(inspect.getsource(distance_transform_edt))
PY
