"""Reuse verified exact CUDA EDT; host-backed full-axis batches handle shared-VRAM volumes."""
import contextlib
import ctypes
import fcntl
import math
import os
import sys
import time
from common import R, MV2, atomic, read, sha

def line_batches(shape, axis, budget):
    """Each work item includes the FULL axis; other-axis batching never clips distances."""
    d = shape[axis]
    count = math.prod(shape) // d
    chunk = max(1, budget // (28*d + 8))
    for first in range(0, count, chunk):
        yield first, min(count, first+chunk)

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

def verify(cp):
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    rows = []
    for shape, spacing, budget in [((7, 11, 19), (30, 8, 8), 5000), ((13, 29), (15, 5), 3000),
                                  ((3, 7, 4096), (40, 12, 12), 2 << 20),
                                  ((3, 128, 4096), (40, 12, 12), 8 << 20)]:
        rng = np.random.default_rng(20261022)
        a = rng.random(shape) > .03
        a.flat[0] = False
        cpu = distance_transform_edt(a, sampling=spacing)
        gpu = host_backed_edt(a, spacing, cp, budget)
        error = float(np.max(np.abs(cpu-gpu)))
        assert np.allclose(cpu, gpu, atol=1e-9, rtol=1e-12), error
        rows.append({'shape': shape, 'spacing': spacing, 'max_abs_error': error, 'workspace_bytes': budget})
    return rows

@contextlib.contextmanager
def accelerated(path):
    enabled = R / 'GPU_SCORING.json'
    if not enabled.exists():
        yield None
        return
    config = read(enabled)
    assert config['enabled'] and config['physical_gpus'] == [1, 2, 3]
    import supervision
    j = read(R / 'supervision.json')
    assert not j['released'] and supervision.identity(j['controller_pid'], j['created'])
    handle, selected = None, None
    for index in config['physical_gpus']:
        candidate = (R / 'locks' / f'gpu_score_{index}.lock').open('a')
        try:
            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle, selected = candidate, index
            break
        except BlockingIOError:
            candidate.close()
    if handle is None:
        raise RuntimeError('Only three registered scorer workers may acquire GPUs')
    try:
        uuid = config['uuids'][str(selected)]
        assert uuid != config['gpu0_uuid']
        # Parent launches scorers with CUDA hidden. Select a physical allowed GPU before torch import.
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        os.environ['MV2_GPU_EDT'] = '1'
        os.environ.setdefault('CUDA_PATH', str(MV2 / 'gpu_deps/nvidia/cuda_runtime'))
        sys.path[:0] = [str(MV2 / 'src'), str(MV2 / 'gpu_deps')]
        nvrtc = MV2 / 'gpu_deps/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
        if nvrtc.exists():
            ctypes.CDLL(str(nvrtc), mode=ctypes.RTLD_GLOBAL)
        import cupy as cp
        import gpu_edt
        original = gpu_edt.edt_array
        checks = verify(cp)
        info = {'physical_gpu': selected, 'uuid': uuid, 'pid': os.getpid(), 'path': str(path),
                'started_at': time.time(), 'equivalence': checks, 'full_device_calls': 0,
                'host_backed_calls': 0, 'source_sha256': sha(__file__)}
        receipt = R / 'evidence' / f'gpu_score_{os.getpid()}.json'

        # Verify the shared MONAI metric path with both exact backends on an anisotropic fixture.
        import numpy as np
        import validation
        fixture = np.zeros((9, 19, 23), bool)
        fixture[2:6, 4:13, 7:16] = True
        shifted = np.roll(fixture, 2, axis=2)
        valid = np.ones(fixture.shape, bool)
        os.environ['MV2_GPU_EDT'] = '0'
        cpu_metric = validation.distances(shifted, fixture, valid, (30., 8., 8.))
        os.environ['MV2_GPU_EDT'] = '1'
        gpu_edt.edt_array = lambda a, sampling=None: host_backed_edt(a, sampling, cp, 5000)
        gpu_metric = validation.distances(shifted, fixture, valid, (30., 8., 8.))
        gpu_edt.edt_array = original
        assert np.allclose(cpu_metric, gpu_metric, atol=1e-6, rtol=1e-10), (cpu_metric, gpu_metric)
        info['MONAI_HD95_NSD_equivalence'] = {'CPU': cpu_metric, 'host_GPU': gpu_metric}

        def edt(array, sampling=None):
            free, _ = cp.cuda.runtime.memGetInfo()
            if 9*array.size + (1 << 30) < min(free-(2 << 30), 8 << 30):
                info['full_device_calls'] += 1
                atomic(receipt, info)
                return original(array, sampling)
            info['host_backed_calls'] += 1
            atomic(receipt, info)
            return host_backed_edt(array, sampling, cp)

        gpu_edt.edt_array = edt
        atomic(receipt, info)
        try:
            yield info
            info.update(completed_at=time.time(), status='complete')
            atomic(receipt, info)
        finally:
            gpu_edt.edt_array = original
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        handle.close()
