from pathlib import Path
import ctypes
import subprocess

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

from heads_model import CONFIG, OUT
from targets import encode


# Diffusion on the foreground graph. Masked neighbours absorb heat at the
# zero-valued boundary. A separate source is used for each visible fragment.
DIFFUSION = r"""
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
extern "C" void heat_flows(int n, int steps, int seed, const int* neighbours, float* output) {
    std::vector<float> a(n+1,0.f), b(n+1,0.f);
    for (int t=0;t<steps;++t) {
        a[seed]+=1.f;
        for (int k=0;k<n;++k) {
            const int* j=neighbours+k*9;
            b[k]=(a[j[0]]+a[j[1]]+a[j[2]]+a[j[3]]+a[j[4]]+a[j[5]]+a[j[6]]+a[j[7]]+a[j[8]])/9.f;
        }
        std::swap(a,b);
    }
    for (int k=0;k<n;++k) {
        const int* j=neighbours+k*9;
        float dy=a[j[7]]-a[j[1]], dx=a[j[5]]-a[j[3]];
        float norm=std::sqrt(dy*dy+dx*dx);
        if (norm>1.e-12f) { output[k]=dy/norm; output[n+k]=dx/norm; }
    }
}
"""
KERNEL = None
NEIGHBOURS = np.array([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)])


def init_diffusion():
    global KERNEL
    if KERNEL is not None:
        return
    path = OUT / "ank_diffusion.so"
    if not path.is_file():
        src = OUT / "ank_diffusion.cpp"
        src.write_text(DIFFUSION)
        tmp = path.with_suffix(".tmp")
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(tmp)], check=True)
        tmp.replace(path)
    lib = ctypes.CDLL(str(path))
    KERNEL = lib.heat_flows
    KERNEL.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
                       np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")]
    KERNEL.restype = None


def centroids(ids):
    sigma = CONFIG["panoptic"]["sigma"]
    out = np.zeros((3, *ids.shape), np.float32)
    slices = ndi.find_objects(np.maximum(ids, 0))
    for k, box in enumerate(slices, 1):
        if box is None:
            continue
        y, x = np.nonzero(ids[box] == k)
        y, x = y + box[0].start, x + box[1].start
        cy, cx = int(np.rint(y.mean())), int(np.rint(x.mean()))
        out[1, y, x] = cy-y
        out[2, y, x] = cx-x
        cut = 3*sigma
        y0, y1 = max(0, cy-cut), min(ids.shape[0], cy+cut+1)
        x0, x1 = max(0, cx-cut), min(ids.shape[1], cx+cut+1)
        grid_y, grid_x = np.ogrid[y0:y1,x0:x1]
        patch = np.exp(-((grid_y-cy)**2+(grid_x-cx)**2)/(2*sigma*sigma))
        out[0,y0:y1,x0:x1] = np.maximum(out[0,y0:y1,x0:x1], patch)
    return out


def flows(ids):
    if KERNEL is None:
        raise RuntimeError("Call init_diffusion before preparing flow targets")
    out = np.zeros((2, *ids.shape), np.float32)
    for k, box in enumerate(ndi.find_objects(np.maximum(ids, 0)), 1):
        if box is None:
            continue
        parts, count = ndi.label(ids[box] == k, structure=np.ones((3,3), bool))
        for part in range(1, count+1):
            y, x = np.nonzero(parts == part)
            n = len(y)
            y, x = y-y.min()+1, x-x.min()+1
            all_y, all_x = np.nonzero(parts == part)
            h, w = int(y.max()+2), int(x.max()+2)
            index = np.full((h,w), n, dtype=np.int32)
            index[y,x] = np.arange(n, dtype=np.int32)
            neighbours = np.ascontiguousarray(index[y[:,None]+NEIGHBOURS[:,0],
                                                       x[:,None]+NEIGHBOURS[:,1]])
            seed = int(np.argmin((y-y.mean())**2+(x-x.mean())**2))
            v = np.zeros((2,n), dtype=np.float32)
            KERNEL(n, 2*(h+w), seed, neighbours, v)
            out[:,all_y+box[0].start,all_x+box[1].start] = v
    return out


def instance_target(arm, ids):
    if arm == "panoptic":
        return centroids(ids)
    if arm == "cellpose":
        return flows(ids)
    return np.asarray(encode(ids), dtype=np.float32)


def smoke_preview(arm, examples, output):
    # Validate target semantics on actual labelled crops before recording previews.
    output.mkdir(exist_ok=True)
    for idx, raw, ids, target in examples:
        fg = ids > 0
        assert fg.any() and np.isfinite(target).all()
        if arm == "panoptic":
            assert target[0].max() == 1 and np.all(target[1:,~fg] == 0)
            for k in np.unique(ids[fg]):
                y,x = np.nonzero(ids == k)
                np.testing.assert_array_equal(y + target[1,y,x], np.full(y.shape, np.rint(y.mean())))
                np.testing.assert_array_equal(x + target[2,y,x], np.full(x.shape, np.rint(x.mean())))
        elif arm == "cellpose":
            norm = np.linalg.norm(target, axis=0)
            assert np.all(norm[~fg] == 0) and np.any(norm[fg] > .99)
            assert np.all((norm < 1e-6) | (np.abs(norm-1) < 1e-5))
        else:
            assert target[1,fg].max() > 0 and np.all(target[1,~fg] == 0)
            assert target[0].any()
        ids_rgb = np.stack([ids*53%255,ids*113%255,ids*159%255],-1)
        ids_rgb[~fg] = 0
        panels = [np.repeat(raw[...,None],3,axis=-1), ids_rgb.astype(np.uint8)]
        for t in target:
            maximum = max(float(abs(t).max()),1e-6)
            scale = np.clip((t/maximum*.5+.5) if t.min()<0 else t/maximum,0,1)
            layer = (scale*255).astype(np.uint8)
            panels.append(np.repeat(layer[...,None],3,axis=-1))
        Image.fromarray(np.concatenate(panels, axis=1)).save(output/f"{idx}.png")
