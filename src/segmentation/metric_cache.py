"""Per-input memoization of official MONAI edge distances; no metric formula changes.

Only installed in the isolated CPU scoring process. Original metric functions
remain the entrypoints; temporary module bindings are restored after each call.
Input and output snapshots protect against mutation and accidental cache reuse.
"""
import importlib,inspect
import torch

def clone_tree(value):
 if isinstance(value,torch.Tensor):return value.clone()
 if isinstance(value,tuple):return tuple(clone_tree(x) for x in value)
 return value

class MetricCache:
 def __init__(self):
  self.modules=[importlib.import_module('monai.metrics.hausdorff_distance'),importlib.import_module('monai.metrics.surface_dice')]
  self.helper=self.modules[0].get_edge_surface_distance
  assert all(m.get_edge_surface_distance is self.helper for m in self.modules)
  self.signature=inspect.signature(self.helper);self.owner=None;self.entries={};self.hits=0;self.misses=0
 def helper_cached(self,*args,**kwargs):
  values=self.signature.bind(*args,**kwargs);values.apply_defaults();v=values.arguments
  # get_mask_edges uses label_idx=1: bool and {0,1} float inputs have the same edges.
  # Hausdorff converts masks to float; surface Dice retains bool. Cache their shared binary semantics.
  p,g=v['y_pred']==1,v['y']==1;spacing=v['spacing']
  if spacing is not None:
   spacing=float(spacing) if isinstance(spacing,(int,float)) else tuple(map(float,spacing))
  key=(tuple(p.shape),p.device,tuple(g.shape),g.device,v['distance_metric'],spacing,v['use_subvoxels'],v['symmetric'],v['class_index'])
  entry=self.entries.get(key)
  if entry is not None and torch.equal(p,entry[0]) and torch.equal(g,entry[1]):
   self.hits+=1;return clone_tree(entry[2])
  result=self.helper(*args,**kwargs);self.misses+=1
  self.entries[key]=(p.clone(),g.clone(),clone_tree(result))
  return result
 def wrap(self,fn):
  def measure(y_pred,y,*args,**kwargs):
   if self.owner is None or self.owner[0] is not y_pred or self.owner[1] is not y:
    self.entries.clear();self.owner=(y_pred,y)
   original=[m.get_edge_surface_distance for m in self.modules]
   try:
    for m in self.modules:m.get_edge_surface_distance=self.helper_cached
    return fn(y_pred,y,*args,**kwargs)
   finally:
    for m,value in zip(self.modules,original):m.get_edge_surface_distance=value
  return measure

def install(hd,nsd):
 cache=MetricCache()
 return cache.wrap(hd),cache.wrap(nsd),cache
