"""Evaluate one validation-locked segmentation head on complete original-grid volumes."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path('/opt/data/3_117/tanghr')
RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'codex/mitotopo_20260908_round2/deps'))
sys.path.insert(0,str(ROOT/'DINOv3/dinov3'))

import torch
from seg_eval import evaluate
from seg_model import SegModel,load_encoder


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',choices=['validation','legacy_dev_test'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    ckpath=args.checkpoint.resolve(strict=True)
    if args.split=='legacy_dev_test':
        lock=RUN/'evidence/validation_selection.lock.json'
        if not lock.exists():
            raise SystemExit('Legacy development-test evaluation requires frozen validation selection')
        selected=json.loads(lock.read_text())
        if sha(ckpath) not in selected['allowed_checkpoint_sha256']:
            raise SystemExit('Checkpoint not among validation-locked selections')
    checkpoint=torch.load(ckpath,map_location='cpu',mmap=True,weights_only=False)
    identity=checkpoint['identity']
    encoder_path=Path(identity['encoder'])
    if sha(encoder_path)!=identity['encoder_sha256']:
        raise SystemExit('Encoder weight hash mismatch')
    model=SegModel(load_encoder(ROOT/'DINOv3/dinov3',encoder_path),identity['seed'],
                   identity['fine_tune_last4']).cuda()
    model.decoder.load_state_dict(checkpoint['decoder'],strict=True)
    if identity['fine_tune_last4']:
        missing,unexpected=model.encoder.load_state_dict(checkpoint['encoder_last4'],strict=False)
        if unexpected or any(any(k.startswith(f'blocks.{i}.') for i in range(20,24)) for k in missing):
            raise SystemExit('Fine-tuned encoder state mismatch')
    report=evaluate(model,RUN/f"data/{args.split}.json",identity['mode'])
    report.update({'split':args.split,'checkpoint':str(ckpath),
                   'checkpoint_sha256':sha(ckpath),'selected_step':checkpoint['completed_steps'],
                   'identity':identity})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if os.environ.get('RUN_OUTPUT'):
        job=Path(os.environ['RUN_OUTPUT'])
        (job/'result.json').write_text(json.dumps({'status':'complete','split':args.split,
            'source_macro_dice':report['source_macro_dice'],
            'source_macro_iou':report['source_macro_iou'],
            'volumes':len(report['volume_rows']),
            'checkpoint_sha256':report['checkpoint_sha256']},indent=2)+'\n')
    print(json.dumps({'status':'complete','split':args.split,
                      'source_macro_dice':report['source_macro_dice'],
                      'volumes':len(report['volume_rows'])}))


if __name__=='__main__':
    main()
