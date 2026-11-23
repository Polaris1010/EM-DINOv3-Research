import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

CONNECTIVITY = np.ones((3, 3), dtype=bool)


def boundary_watershed(fg, bnd, t_fg, t_bnd, min_size):
    foreground = fg > t_fg
    markers, _ = ndi.label(foreground & (bnd <= t_bnd), structure=CONNECTIVITY)
    counts = np.bincount(markers.ravel())
    keep = counts >= min_size
    keep[0] = False
    if not keep.any():
        return np.zeros(foreground.shape, dtype=np.int32)
    remap = np.cumsum(keep, dtype=np.int32) * keep
    return watershed(bnd, remap[markers], mask=foreground, connectivity=CONNECTIVITY)
