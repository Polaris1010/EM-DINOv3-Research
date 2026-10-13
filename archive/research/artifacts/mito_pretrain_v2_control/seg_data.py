"""Deterministic source-balanced supervised 2D and central-slice 2.5D crops."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

SIZE = 256
MEAN = np.array((.485, .456, .406), dtype=np.float32)[:, None, None]
STD = np.array((.229, .224, .225), dtype=np.float32)[:, None, None]


class SegCrops(Dataset):
    def __init__(self, manifest: Path, seed: int, start: int, count: int, mode: str = '2d'):
        if mode not in {'2d', '2p5d'}:
            raise ValueError(mode)
        self.seed, self.start, self.count, self.mode = seed, start, count, mode
        rows = json.loads(manifest.read_text(encoding='utf-8'))
        if not rows or any(row['split'] != 'train' for row in rows):
            raise ValueError('Segmentation crop manifest must contain train rows only')
        if any(row['original_dataset_id'] == 'han24' for row in rows):
            raise ValueError('Ambiguous han24 valid masks are excluded from supervised training')
        self.groups = defaultdict(list)
        for row in rows:
            self.groups[row['original_dataset_id']].append(row)
        self.sources = sorted(self.groups)
        self.handles = {}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        absolute = self.start + index
        rng = np.random.default_rng(self.seed * 1_000_000_007 + absolute + 31_337)
        source = self.sources[int(rng.integers(len(self.sources)))]
        row = self.groups[source][int(rng.integers(len(self.groups[source])))]
        vid = row['volume_id']
        if vid not in self.handles:
            if len(self.handles) >= 2:
                self.handles.clear()
            self.handles[vid] = zarr.open_group(row['local_path'], mode='r')
        group = self.handles[vid]
        img, mito = group['img'], group['mito']
        mask = group['mask'] if 'mask' in group else None
        depth, height, width = img.shape
        demand_fg = bool(rng.integers(2))
        for attempt in range(128):
            z = int(rng.integers(depth))
            y = int(rng.integers(max(1, height - SIZE + 1)))
            x = int(rng.integers(max(1, width - SIZE + 1)))
            sl = (z, slice(y, y+SIZE), slice(x, x+SIZE))
            label = np.asarray(mito[sl])
            valid = np.asarray(mask[sl]) != 0 if mask is not None else np.ones(label.shape, bool)
            target = np.where(valid, label > 0, -1).astype(np.int64)
            if not valid.any() or (demand_fg and not (target == 1).any()):
                continue
            break
        else:
            raise RuntimeError(f'No eligible {"foreground" if demand_fg else "valid"} patch for {vid}')
        slices = [z] if self.mode == '2d' else [max(0, z-1), z, min(depth-1, z+1)]
        planes = [np.asarray(img[zi, y:y+SIZE, x:x+SIZE], dtype=np.uint8) for zi in slices]
        py, px = SIZE-target.shape[0], SIZE-target.shape[1]
        if py or px:
            planes = [np.pad(plane, ((0,py),(0,px)), mode='reflect') for plane in planes]
            target = np.pad(target, ((0,py),(0,px)), constant_values=-1)
        if rng.integers(2):
            planes = [np.flip(plane, axis=0) for plane in planes]
            target = np.flip(target, axis=0)
        if rng.integers(2):
            planes = [np.flip(plane, axis=1) for plane in planes]
            target = np.flip(target, axis=1)
        turns = int(rng.integers(4))
        planes = [np.rot90(plane, turns) for plane in planes]
        target = np.rot90(target, turns)
        if self.mode == '2d':
            image = np.repeat(planes[0][None], 3, axis=0)
        else:
            image = np.stack(planes)
        image = (image.astype(np.float32)/255.0 - MEAN)/STD
        location = {'index': absolute, 'source': source, 'volume_id': vid,
                    'z': z, 'y': y, 'x': x, 'force_foreground': demand_fg,
                    'attempt': attempt, 'mode': self.mode}
        return torch.from_numpy(image.copy()), torch.from_numpy(target.copy()), location


def collate(samples):
    locations = [item[2] for item in samples]
    paired_locations = [{key:value for key,value in location.items() if key != 'mode'}
                        for location in locations]
    digest = hashlib.sha256(json.dumps(paired_locations, sort_keys=True).encode()).hexdigest()
    return {'image': torch.stack([item[0] for item in samples]),
            'target': torch.stack([item[1] for item in samples]),
            'locations': locations, 'sample_digest': digest}
