"""One fixed full legacy-dev evaluation of each validation-selected corrected head."""
import os,sys,json,time,argparse
from pathlib import Path
from settings import *
sys.path.insert(0,str(R4))
import common as c, train as old
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();out=Path(os.environ['RUN_OUTPUT'])
    protocol=json.loads((BASE/'exploratory_baseline_protocol.json').read_text());assert c.sha(P1)==protocol['encoder_sha256']
    ck=torch.load(BASE/'jobs'/f'masked_P1_seed{a.seed}_10000'/'head_best.pth',map_location='cpu',weights_only=False)
    assert ck['identity']['seed']==a.seed and ck['identity']['encoder_sha256']==protocol['encoder_sha256']
    assert ck['identity']['schedule_sha256']==c.sha(BASE/f'train_patch_schedule_masked_seed{a.seed}.jsonl')
    split=json.loads((R4/'split_mitoverse_probe_v1.json').read_text())
    for sp in ['test1','test2','test3']:
        for vid in split[sp]:assert not (Path(split['volumes'][vid]['image_path']).parent/'mask').exists()
    c.seed(a.seed);net=c.model('P1',a.seed);c.load_head(net,ck['head']);t=time.time()
    result=old.evaluate(net,'N2',out,['test1','test2','test3'],True)
    for row in result['volumes'].values():row['split']='legacy_dev_test'
    result.update(status='complete',exploratory=True,selected_step=ck['step'],seed=a.seed,seconds=time.time()-t,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),head_checkpoint_sha256=c.sha(BASE/'jobs'/f'masked_P1_seed{a.seed}_10000'/'head_best.pth'))
    c.save(out/'metrics.json',result);c.save(out/'result.json',result)

if __name__=='__main__':main()
