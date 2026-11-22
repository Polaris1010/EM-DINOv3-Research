"""GPU integration test on the first fixed held-out CEM image and synthetic touching IDs."""
from pathlib import Path
import numpy as np
import tifffile
from model import build, read, targets, torch, OUT
from mixture import C, ROOT, atomic
from inference import predict
from evaluate_head import example, touching, score, ndimage

def test_interface():
    net,probe,stage,head,_=build()
    state=torch.load(OUT/"resume.pt",map_location="cpu",weights_only=True)
    head.load_state_dict(state["head"])
    head.eval()
    manifest=read(C/"manifest.json")
    group=manifest["validation"][0]
    name=manifest["groups"][group][0][0]
    a,b=np.load(Path("/opt/cem_mitolab_20261118")/group/(name+".npy"))
    gt=tifffile.imread(ROOT/"MitoNet/dataset/EMPIAR-11037/cem_mitolab"/group/"masks"/name)
    assert np.array_equal(gt>0,b>0)
    ref=probe.sliding(net,a,tile_batch_size=16)
    sem,ids,fallback=predict(net,probe,stage,head,a)
    assert np.array_equal(sem,ref) and np.array_equal(ids>0,sem)
    base=ndimage.label(ref,np.ones((3,3),int))[0]
    example(OUT/"interface_example",a,gt,sem,ids,base)
    synthetic=torch.zeros((64,64),device="cuda",dtype=torch.long)
    synthetic[0:10,5:15]=12
    synthetic[20:30,20:30]=55
    synthetic[30:40,20:30]=83
    synthetic[-1]=-1
    heat,offset,valid,fg=targets(synthetic)
    assert not valid[0:10,5:15].any() and not valid[-1].any()
    assert fg.sum()==200 and (heat==1).sum()==2
    assert torch.allclose(offset[:,20,20]*32,offset.new_tensor([4.5,4.5]))
    lab=synthetic.clamp_min(0).cpu().numpy()
    assert touching(lab,lab)==(1,1)
    atomic(OUT/"interface_test.json",dict(step=state["step"],image=name,shape=list(a.shape),
           differing_pixels=int(np.count_nonzero(sem!=ref)),fallback=fallback,head=score(gt,ids),cc=score(gt,base)))

if __name__=="__main__":
    test_interface()
