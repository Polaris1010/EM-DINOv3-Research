"""Exact CUDA EDT, reusing the verified D12 full-axis host-backed algorithm."""
import contextlib
import ctypes
import math
import os
import subprocess
import sys
import time
from system_common import R, MV2, atomic, sha

def axis_tiles(shape, axis, budget):
    """Rectangular tiles keep a full transform axis and contiguous host inner rows."""
    import itertools
    capacity = max(1, budget // (28*shape[axis]+8))
    widths = [1]*len(shape)
    widths[axis] = shape[axis]
    for dim in reversed(range(len(shape))):
        if dim != axis:
            widths[dim] = min(shape[dim], capacity)
            capacity = max(1, capacity//widths[dim])
    starts = [range(0, n, w) for n,w in zip(shape,widths)]
    for first in itertools.product(*starts):
        yield tuple(slice(a, min(n,a+w)) for a,n,w in zip(first,shape,widths))

def host_backed_edt(array, spacing, cp, workspace_bytes=256 << 20):
    import numpy as np
    from separable_edt import SOURCE
    module = cp.RawModule(code=SOURCE, options=('--fmad=false',), name_expressions=('initialize', 'transform'))
    transform = module.get_function('transform')
    shape = tuple(array.shape)
    spacing = [1.]*len(shape) if spacing is None else list(map(float, spacing))
    assert len(shape) in (2, 3) and len(spacing) == len(shape)
    assert all(math.isfinite(s) and s > 0 for s in spacing)
    if not np.any(~array):
        # Retain SciPy's implicit-boundary convention for the degenerate no-site case.
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(array, sampling=spacing)
    distance = np.where(array, np.inf, 0.)
    for axis, d in enumerate(shape):
        for tile in axis_tiles(shape, axis, workspace_bytes):
            # Transfer contiguous host row blocks; perform the axis transpose on CUDA.
            # All complete lines are still independent in this one-dimensional pass.
            raw = cp.asarray(np.ascontiguousarray(distance[tile]))
            values = cp.ascontiguousarray(cp.moveaxis(raw, axis, -1))
            del raw
            count = values.size//d
            sites = cp.empty((count, d), dtype=cp.int32)
            cuts = cp.empty((count, d+1), dtype=cp.float64)
            costs = cp.empty((count, d), dtype=cp.float64)
            transform(((count+127)//128,), (128,), (values, np.int32(d), np.int64(1),
                      np.int64(0), np.int64(count), np.float64(spacing[axis]**2), sites, cuts, costs))
            del sites, cuts, costs
            restored = cp.ascontiguousarray(cp.moveaxis(values, -1, axis))
            del values
            distance[tile] = cp.asnumpy(restored)
            del restored
            cp.get_default_memory_pool().free_all_blocks()
    np.sqrt(distance, out=distance)
    return distance


@contextlib.contextmanager
def accelerated(label):
    """Use only the controller-assigned physicalGPU1/2/3; never silently use CPU."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {index.strip(): uuid.strip() for index, uuid in (line.split(',') for line in gpu_rows.splitlines())}
    selected = next((int(index) for index, uuid in mapping.items() if visible in (index, uuid)), None)
    assert selected in (1, 2, 3), 'A single physicalGPU1/2/3 must be assigned'
    os.environ['CUDA_VISIBLE_DEVICES'] = mapping[str(selected)]
    os.environ['MV2_GPU_EDT'] = '1'
    os.environ.setdefault('CUDA_PATH', str(MV2 / 'gpu_deps/nvidia/cuda_runtime'))
    sys.path[:0] = [str(MV2 / 'src'), str(MV2 / 'gpu_deps')]
    nvrtc = MV2 / 'gpu_deps/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
    if nvrtc.exists():
        ctypes.CDLL(str(nvrtc), mode=ctypes.RTLD_GLOBAL)
    import cupy as cp
    import gpu_edt
    original = gpu_edt.edt_array
    info = {'physical_gpu': selected, 'uuid': mapping[str(selected)], 'label': str(label),
            'source_sha256': sha(__file__), 'full_device_calls': 0, 'host_backed_calls': 0,
            'started_at': time.time(), 'status': 'running'}
    receipt = R / 'evidence' / f'gpu_scoring_{os.getpid()}.json'
    def edt(array, sampling=None):
        free, _ = cp.cuda.runtime.memGetInfo()
        if 9*array.size + (1 << 30) < min(free-(2 << 30), 8 << 30):
            info['full_device_calls'] += 1
            return original(array, sampling)
        info['host_backed_calls'] += 1
        return host_backed_edt(array, sampling, cp)
    gpu_edt.edt_array = edt
    import validation
    original_distances = validation.distances
    def distances(pred, gt, valid, spacing):
        if valid.all():
            return original_distances(pred, gt, valid, spacing)
        import numpy as np
        from scipy.ndimage import binary_erosion
        from monai.metrics.utils import get_mask_edges
        e1, e2 = get_mask_edges(pred, gt, crop=False, always_return_as_numpy=True)
        safe = binary_erosion(valid, border_value=1)
        e1 &= safe
        e2 &= safe
        if not e1.any() or not e2.any():
            return math.inf, 0.
        d = edt(~e2, spacing)
        a = d[e1].copy()
        del d
        d = edt(~e1, spacing)
        b = d[e2].copy()
        del d
        return max(float(np.percentile(a, 95)), float(np.percentile(b, 95))), float(((a<=16).sum()+(b<=16).sum())/(len(a)+len(b)))
    validation.distances = distances
    atomic(receipt, info)
    try:
        yield info
        info.update(completed_at=time.time(), status='complete')
    except BaseException as error:
        info.update(completed_at=time.time(), status='failed', error=repr(error))
        raise
    finally:
        gpu_edt.edt_array = original
        validation.distances = original_distances
        atomic(receipt, info)
        cp.get_default_memory_pool().free_all_blocks()
