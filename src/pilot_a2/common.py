"""Pilot A2 immutable inputs and isolated outputs; no model training."""
import csv,hashlib,importlib.util,json,os,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
R=Path(os.environ['PILOT_A2_ROOT'])
P=Path('/opt/data/3_117/tanghr/codex/feature_mechanism_pilot_a_20261027')
MODELS=['D2-30k','D2-100k']
SOURCES=['openorganelle','guay21','muller21','glycolytic_muscle','hela_cell']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def csvout(name,rows):
    p=R/'tables'/name;p.parent.mkdir(exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def parent_data():
    spec=importlib.util.spec_from_file_location('pilot_parent_io',P/'code/common.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.Data()
def samples():return read(P/'samples.json')
def native(model,ids):
    a=np.load(P/'models'/model/'linear_probability.npy',mmap_mode='r')
    z=torch.tensor(np.array(a[ids,3],copy=True),device='cuda').reshape(-1,1,16,16)
    return F.interpolate(z,size=(256,256),mode='bilinear',align_corners=False)[:,0]
def setup():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.6)
    assert torch.cuda.is_available()
def counts(p,g,v,far,b):
    pd=p.double();t=1/(1+math.exp(b));pred=pd>t
    eb=math.exp(b);q=eb*pd/(1-pd+eb*pd)
    assert torch.equal(pred,q>.5),'Bias transform and threshold disagree'
    return {'tp':int((pred&g&v).sum()),'fp':int((pred&~g&v).sum()),'fn':int((~pred&g&v).sum()),
      'valid_pixels':int(v.sum()),'foreground_pixels':int((g&v).sum()),'predicted_pixels':int((pred&v).sum()),
      'far_background_pixels':int(far.sum()),'far_FP_pixels':int((pred&far).sum()),'brier_sum':float(((q-g.double()).square()*v).sum())}
def summarize(c):
    tp,fp,fn=c['tp'],c['fp'],c['fn'];f=c['far_background_pixels'];n=c['valid_pixels']
    return {'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,
      'recall':tp/(tp+fn) if tp+fn else None,'far_FP_rate':c['far_FP_pixels']/f if f else None,
      'predicted_foreground_fraction':c['predicted_pixels']/n,'brier':c['brier_sum']/n}
class Ranked:
    """Exact tie-aware ROC/AP; one GPU sort reused for clustered bootstrap."""
    def __init__(self,score,target,block):
        order=torch.argsort(score,descending=True,stable=True)
        self.score=score[order];self.y=target[order].long();self.block=block[order]
        self.ends=torch.cat([torch.nonzero(self.score[1:]!=self.score[:-1],as_tuple=False).flatten(),torch.tensor([len(score)-1],device='cuda')])
    def calculate(self,block_weights=None,curve=False):
        w=torch.ones_like(self.y) if block_weights is None else block_weights[self.block]
        tp=(w*self.y).cumsum(0)[self.ends].double();total=w.cumsum(0)[self.ends].double();fp=total-tp
        if float(tp[-1])==0 or float(fp[-1])==0:return None,None,None
        recall=tp/tp[-1];fpr=fp/fp[-1];precision=tp/total
        dr=torch.diff(recall,prepend=recall.new_zeros(1));df=torch.diff(fpr,prepend=fpr.new_zeros(1))
        auroc=(df*(recall+torch.cat([recall.new_zeros(1),recall[:-1]]))*.5).sum();ap=(dr*precision).sum();out=None
        if curve:
            grid=torch.linspace(0,1,201,device='cuda',dtype=torch.float64)
            idx=torch.searchsorted(recall.contiguous(),grid).clamp_max(len(recall)-1)
            out=torch.stack([recall[idx],precision[idx],fpr[idx]],1).cpu().tolist()
        return float(auroc),float(ap),out
