"""Bounded native-data, image-only loader smoke; no GT or model selection."""
import collections,time
import torch
from omegaconf import OmegaConf
from dinov3.configs import get_default_config
from common import R,read,atomic,sha
from ssl_full_data import Images,Collate
from sampling import choose
def main():
 t=time.time();cfg=OmegaConf.merge(get_default_config(),OmegaConf.load(R/'D12_base.yaml'));results={}
 for model in ['D1','D2']:
  spec=R/(model+'_manifest.json');data=Images(spec,cfg,0,0,200);shapes=[];domains=collections.Counter()
  for i in range(32):
   image,where=data.raw(i);shapes.append(list(image.size));domains[where['domain']]+=1
  batch=Collate(cfg)([data[i] for i in range(16)])
  assert batch['collated_global_crops'].shape==(32,3,256,256) and batch['collated_local_crops'].shape==(128,3,112,112)
  assert torch.isfinite(batch['collated_global_crops']).all() and torch.isfinite(batch['collated_local_crops']).all()
  draws=collections.Counter();sources=collections.Counter()
  for i in range(20000):
   w=choose(data.pool,data.cem,data.spec['cem_fraction'],0,i);draws[w['domain']]+=1;sources[w['domain']+'::'+w['source']]+=1
  if model=='D1':assert draws=={'MitoVerse':20000}
  else:assert 9500<draws['CEM1.5M']<10500
  results[model]={'native_input_wh':shapes,'raw_domains32':dict(domains),'simulated_domains20000':dict(draws),'simulated_sources20000':dict(sources),'global_shape':list(batch['collated_global_crops'].shape),'local_shape':list(batch['collated_local_crops'].shape),'manifest_sha256':sha(spec),'label_reads':False}
 atomic(R/'loader_smoke.json',{'status':'PASS','runs':results,'seconds':time.time()-t});print('LOADER_SMOKE_PASS',time.time()-t,flush=True)
if __name__=='__main__':main()
