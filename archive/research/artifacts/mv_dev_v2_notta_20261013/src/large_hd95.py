"""CPU fallback for MONAI's linear percentile beyond torch.quantile's size limit.

Edges, physical distances, symmetric aggregation and NSD remain in the original
MONAI functions. Only an oversized percentile vector uses NumPy partitioning.
Bindings are scoped to this CPU metric call and restored afterwards.
"""
from contextlib import contextmanager
import importlib
import numpy as np
import torch

MAX_TORCH_QUANTILE_VALUES=1<<24

def percentile(surface_distance,percentile,original):
 if surface_distance.numel()<=MAX_TORCH_QUANTILE_VALUES or percentile is None or not 0<=percentile<=100:
  return original(surface_distance,percentile)
 if surface_distance.device.type!='cpu':raise ValueError('Oversized HD95 fallback requires CPU distances')
 value=np.quantile(surface_distance.detach().numpy(),percentile/100,method='linear')
 return torch.as_tensor(value,dtype=surface_distance.dtype,device=surface_distance.device)

@contextmanager
def compatibility():
 module=importlib.import_module('monai.metrics.hausdorff_distance')
 original=module._compute_percentile_hausdorff_distance
 module._compute_percentile_hausdorff_distance=lambda distances,q:percentile(distances,q,original)
 try:yield
 finally:module._compute_percentile_hausdorff_distance=original
