from common import *
import train
from data import Patches
# Full-grid blending must preserve a constant probability, including padded and overlapping tiles.
old_cuda=torch.Tensor.cuda;old_forward=train.forward
try:
 torch.Tensor.cuda=lambda self,*a,**k:self
 def constant(net,x):
  q=torch.zeros((len(x),2,256,256));q[:,1]=np.log(3.);return q
 train.forward=constant
 checks=[]
 for shape in [(100,100),(256,256),(300,501),(512,768)]:
  p=train.slide(None,np.zeros(shape,np.uint8),'N1');assert p.shape==shape and np.allclose(p,.75,atol=1e-6);checks.append(list(shape))
finally:torch.Tensor.cuda=old_cuda;train.forward=old_forward
assert metrics(confusion(np.ones((4,4),bool),np.ones((4,4),bool)))['dice']==1
assert metrics(confusion(np.zeros((4,4),bool),np.ones((4,4),bool)))['dice']==0
x=Patches(0,16,'N1');a=x[0];b=x[0];assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]);assert a[2]==b[2]==0
save(BASE/'verification_tests.json',{'sliding_constant_probability_preserved_shapes':checks,'metric_perfect_and_empty':'pass','repeat_patch_and_augmentation':'identical','schedule_train_only':'asserted per sample','guard_trends':'tiny swap tolerated; >8GiB sustained RSS stopped; >1GiB swap requires RSS or MemAvailable evidence (tested at launch)'})
print('VERIFICATION_PASS')
