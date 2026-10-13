"""CPU gate for valid-mask targets, paired 2D/2.5D sampling and decoder shape."""

import argparse
import json
from pathlib import Path

import torch

from seg_data import SegCrops, collate
from seg_model import PatchDecode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    a = SegCrops(args.manifest, 0, 0, 2, '2d')
    b = SegCrops(args.manifest, 0, 0, 2, '2p5d')
    two = [a[i] for i in range(2)]
    two_context = [b[i] for i in range(2)]
    assert [{k:v for k,v in x[2].items() if k != 'mode'} for x in two] == [
        {k:v for k,v in x[2].items() if k != 'mode'} for x in two_context]
    assert all(torch.equal(x[1], y[1]) for x,y in zip(two,two_context))
    batch = collate(two)
    assert batch['sample_digest'] == collate(two_context)['sample_digest']
    assert batch['image'].shape == (2,3,256,256)
    assert batch['target'].shape == (2,256,256)
    assert all((target >= 0).any() for target in batch['target'])
    head = PatchDecode().eval()
    with torch.inference_mode():
        logits = head(torch.zeros((1,4096,16,16)))
    assert logits.shape == (1,2,256,256)
    result = {'status':'complete', 'locations':batch['locations'],
              'target_unique':[sorted(map(int,torch.unique(x))) for x in batch['target']],
              'mode_pairing_exact':True, 'decoder_output_shape':list(logits.shape),
              'decoder_parameters':sum(p.numel() for p in head.parameters()),
              'sample_digest':batch['sample_digest']}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
