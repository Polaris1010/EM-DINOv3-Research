"""Full-volume native-grid semantic validation for the locked 2D protocol."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import zarr

from seg_data import MEAN, SIZE, STD


def positions(length: int, stride: int = 128):
    if length <= SIZE:
        return [0]
    result = list(range(0, length-SIZE+1, stride))
    if result[-1] != length-SIZE:
        result.append(length-SIZE)
    return result


def metrics(tp, fp, fn, tn):
    d = 2*tp+fp+fn
    u = tp+fp+fn
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'dice': float(2*tp/d) if d else 1.0,
            'iou': float(tp/u) if u else 1.0,
            'precision': float(tp/(tp+fp)) if tp+fp else (0.0 if fn else 1.0),
            'recall': float(tp/(tp+fn)) if tp+fn else 1.0}


def tile_inputs(planes, y, x):
    patches = []
    for plane in planes:
        crop = plane[y:y+SIZE, x:x+SIZE]
        py, px = SIZE-crop.shape[0], SIZE-crop.shape[1]
        if py or px:
            crop = np.pad(crop, ((0,py),(0,px)), mode='reflect')
        patches.append(crop)
    image = np.repeat(patches[0][None], 3, axis=0) if len(patches) == 1 else np.stack(patches)
    return (image.astype(np.float32)/255.0-MEAN)/STD


@torch.inference_mode()
def evaluate(model, manifest: Path, mode: str = '2d', max_volumes: int | None = None):
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    if any(row['split'] not in {'validation','legacy_dev_test'} for row in rows):
        raise ValueError('Evaluation manifest contains training data')
    if max_volumes is not None:
        rows = rows[:max_volumes]
    source = defaultdict(list)
    volume_rows = []
    device = next(model.parameters()).device
    axis = np.arange(SIZE, dtype=np.float32) - (SIZE-1)/2
    weight = np.exp(-.5*(axis[:,None]**2+axis[None,:]**2)/(SIZE/4)**2).astype(np.float32)
    weight = np.maximum(weight, 1e-3)
    model.eval()
    for row in rows:
        group = zarr.open_group(row['local_path'], mode='r')
        img, labels = group['img'], group['mito']
        region = group['mask'] if 'mask' in group else None
        depth, height, width = img.shape
        counts = np.zeros(4, dtype=np.int64)
        for z in range(depth):
            zs = [z] if mode == '2d' else [max(0,z-1),z,min(depth-1,z+1)]
            planes = [np.asarray(img[zi], dtype=np.uint8) for zi in zs]
            accumulator = np.zeros((height,width), dtype=np.float32)
            denominator = np.zeros((height,width), dtype=np.float32)
            coords = [(y,x) for y in positions(height) for x in positions(width)]
            for offset in range(0, len(coords), 4):
                chunk = coords[offset:offset+4]
                tensor = torch.from_numpy(np.stack([tile_inputs(planes,y,x) for y,x in chunk])).to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    probability = model(tensor).float().softmax(dim=1)[:,1]
                probs = probability.cpu().numpy()
                for (y,x), prob in zip(chunk, probs):
                    h, w = min(SIZE,height-y), min(SIZE,width-x)
                    accumulator[y:y+h,x:x+w] += prob[:h,:w]*weight[:h,:w]
                    denominator[y:y+h,x:x+w] += weight[:h,:w]
            prediction = accumulator/np.maximum(denominator, 1e-8) >= .5
            truth = np.asarray(labels[z]) > 0
            valid = np.asarray(region[z]) != 0 if region is not None else np.ones((height,width),bool)
            counts += [int((prediction & truth & valid).sum()),
                       int((prediction & ~truth & valid).sum()),
                       int((~prediction & truth & valid).sum()),
                       int((~prediction & ~truth & valid).sum())]
        item = {'volume_id':row['volume_id'], 'source':row['original_dataset_id'],
                'slices':depth, **metrics(*counts)}
        volume_rows.append(item)
        source[item['source']].append(item)
    source_rows = [{'source':name,'volumes':len(items),
                    'mean_dice':float(np.mean([item['dice'] for item in items])),
                    'mean_iou':float(np.mean([item['iou'] for item in items]))}
                   for name,items in sorted(source.items())]
    return {'volume_rows':volume_rows,'source_rows':source_rows,
            'source_macro_dice':float(np.mean([row['mean_dice'] for row in source_rows])),
            'source_macro_iou':float(np.mean([row['mean_iou'] for row in source_rows])),
            'mode':mode, 'threshold':.5, 'tta':False, 'tile':SIZE, 'stride':128,
            'valid_policy':'mask != 0 when present', 'full_volume':max_volumes is None}
