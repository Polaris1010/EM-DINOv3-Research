"""Independent tiny numerical references; no historical GT/model data."""
import math
import torch
from common import R,Ranked,atomic,counts,setup
from fit import solve_bias
def reference(scores,labels,weights):
    pos=sum(y*w for y,w in zip(labels,weights));neg=sum((1-y)*w for y,w in zip(labels,weights))
    auc=sum(wi*wj*(1 if a>b else .5 if a==b else 0) for a,yi,wi in zip(scores,labels,weights) if yi for b,yj,wj in zip(scores,labels,weights) if not yj)/(pos*neg)
    ap=0;prev=0
    for t in sorted(set(scores),reverse=True):
        tp=sum(y*w for a,y,w in zip(scores,labels,weights) if a>=t);total=sum(w for a,w in zip(scores,weights) if a>=t)
        ap+=(tp-prev)/pos*(tp/total);prev=tp
    return auc,ap
def run():
    setup();s=[.1,.4,.4,.8,.8,.9];y=[0,0,1,0,1,1];blocks=[0,0,1,1,2,2]
    ranked=Ranked(torch.tensor(s,device='cuda'),torch.tensor(y,device='cuda'),torch.tensor(blocks,device='cuda'))
    cases=[]
    for bw in [[1,1,1],[2,0,1],[0,2,1],[3,1,2]]:
        a,b,_=ranked.calculate(torch.tensor(bw,device='cuda'));ref=reference(s,y,[bw[k] for k in blocks])
        assert abs(a-ref[0])<1e-12 and abs(b-ref[1])<1e-12,(a,b,ref);cases.append([a,b])
    z=torch.zeros(10,device='cuda',dtype=torch.float64);target=z.clone();target[0]=1;w=torch.ones_like(z)/10
    bias,res,n=solve_bias(z,target,w);assert abs(bias-math.log(1/9))<1e-8
    p=torch.tensor([0.,.25,.5,.75,1.],device='cuda');g=torch.tensor([0,0,0,1,1],device='cuda',dtype=torch.bool);v=torch.ones_like(g)
    c=counts(p,g,v,~g,-math.log(3));assert c['predicted_pixels']==1
    atomic(R/'evidence/numerical_smoke.json',{'status':'PASS','GPU':True,'weighted_tie_AUROC_AP_cases':cases,'bias_constant_score_closed_form_error':bias-math.log(1/9),'threshold_transform_fixture':True,'historical_inputs_read':False})
if __name__=='__main__':run()
