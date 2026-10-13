import os
assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
import sys,json,hashlib,importlib.util,gc,subprocess,urllib.request
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1]
sys.path[:0]=['/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps']
spec=importlib.util.spec_from_file_location('official_probe',R/'src/frozen_probe_official.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
registry=json.loads((R/'model_registry.json').read_text())['models']
def state_sha(net):
 h=hashlib.sha256()
 for n,t in sorted(net.state_dict().items()):
  h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
  h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def main():
 rows=[]
 for m in registry:
  assert p.digest(m['checkpoint_path'])==m['sha256']
  p.ENCODER_STEP={'P0':0,'B100k':100000,'B150k':150000}[m['id']]
  net=p.model(Path(m['checkpoint_path']),20260907,'official_natural' if m['id']=='P0' else 'B_full')
  assert not hasattr(net,'last_gates') and not any('gate' in n for n,_ in net.named_parameters())
  initial=p.head_digest(net);before=state_sha(net.dino_encoder)
  sample=p.Samples('Dataset111_CMCC',20260907);x,y,case=sample.sample(0)
  net.dino_encoder.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):fs=net.dino_encoder.get_intermediate_layers(x.repeat(1,3,1,1),n=net.interaction_indices,reshape=True)
  params=[v for v in net.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4)
  losses=[]
  for i in range(2):
   opt.zero_grad(set_to_none=True)
   loss=p.loss(net.up_projection(torch.cat([f.float() for f in fs],1)),y)
   assert torch.isfinite(loss);loss.backward()
   assert all(v.grad is None for v in net.dino_encoder.parameters())
   assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
   opt.step();losses.append(float(loss.detach()))
  assert state_sha(net.dino_encoder)==before
  net.eval()
  logits=p.sliding(net,x[0,0].cpu().numpy(),tta=True)
  assert torch.isfinite(torch.from_numpy(logits)).all()
  rows.append({'model':m['id'],'status':'PASS','decoder_class':type(net).__module__+'.'+type(net).__name__,
     'head_initial_sha256':initial,'trainable_parameters':sum(v.numel() for v in params),'gates_present':False,
     'optimizer_updates':2,'losses':losses,'encoder_unchanged':True,'full_forward_fp16_tta_finite':True,
     'case':case,'decoder_source_sha256':p.digest(R/'src/official_primus.py')})
  del net,params,opt,sample,x,y,fs;gc.collect();torch.cuda.empty_cache();print('PASS',m['id'],initial,flush=True)
 assert len({r['head_initial_sha256'] for r in rows})==1
 with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=8) as response:assert response.status==200
 (R/'audit/smoke.json').write_text(json.dumps({'status':'PASS','models':rows,'website_http_after':200},indent=2))
 print('ALL OFFICIAL PRIMUS SMOKES PASS',flush=True)
if __name__=='__main__':main()
