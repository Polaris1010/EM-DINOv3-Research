"""Full-slice-population image-only Zarr reader for vanilla DINOv3 SSL."""
from __future__ import annotations

import bisect
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import Dataset

from dinov3.data import DataAugmentationDINO
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator


def open_image_only(row: dict):
    return zarr.open_group(row['local_path'], mode='r')[row['image_array_path']]


class Images(Dataset):
    """Uniform over legal Z slices; all XY crop starts remain eligible."""

    def __init__(self, manifest: Path, cfg, seed: int, start: int, count: int,
                 image_opener=open_image_only, *, rank: int = 0,
                 world_size: int = 1, batch_per_rank: int | None = None):
        self.start, self.count, self.seed = start, count, seed
        self.rank, self.world_size, self.batch_per_rank = rank, world_size, batch_per_rank
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid distributed sampler rank')
        if world_size > 1 and (not batch_per_rank or start % batch_per_rank):
            raise ValueError('Distributed sampler requires batch-aligned start')
        self.image_opener = image_opener
        self.handles: dict[str, object] = {}
        doc = json.loads(manifest.read_text(encoding='utf-8'))
        rows = doc['volumes']
        if doc['training_role'] != 'ssl_full_mitoverse' or doc['volume_count'] != len(rows) or not rows:
            raise ValueError('Expected full MitoVerse SSL manifest')
        ids = [r['volume_id'] for r in rows]
        paths = [r['local_path'] for r in rows]
        if len(set(ids)) != len(rows) or len(set(paths)) != len(rows):
            raise ValueError('Duplicate volume ID or path')
        if any(r['training_role'] != 'ssl_full_mitoverse' or r['axis_order'] != 'ZYX'
               or r['image_array_path'] != 'img' or len(r['shape']) != 3
               or min(r['shape']) < 1 for r in rows):
            raise ValueError('Invalid image-only manifest row')
        self.rows = rows
        self.cumulative = []
        depth_total = 0
        for row in rows:
            depth_total += int(row['shape'][0])
            self.cumulative.append(depth_total)
        self.slice_count = depth_total
        self.aug = DataAugmentationDINO(
            global_crops_scale=cfg.crops.global_crops_scale,
            local_crops_scale=cfg.crops.local_crops_scale,
            local_crops_number=cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=None,
            gram_teacher_no_distortions=False,
            horizontal_flips=cfg.crops.horizontal_flips,
            teacher_no_color_jitter=False,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
        )

    def __len__(self):
        return self.count

    def raw(self, absolute_index: int):
        rng = np.random.default_rng(self.seed * 1_000_000_007 + absolute_index)
        slice_number = int(rng.integers(self.slice_count))
        row_index = bisect.bisect_right(self.cumulative, slice_number)
        row = self.rows[row_index]
        previous = self.cumulative[row_index - 1] if row_index else 0
        z = slice_number - previous
        vid = row['volume_id']
        if vid not in self.handles:
            if len(self.handles) >= 2:
                self.handles.clear()
            self.handles[vid] = self.image_opener(row)
        array = self.handles[vid]
        if list(array.shape) != row['shape'] or np.dtype(array.dtype) != np.dtype(row['dtype']):
            raise ValueError(f'Image metadata changed: {vid}')
        _, height, width = row['shape']
        y = int(rng.integers(max(1, height - 512 + 1)))
        x = int(rng.integers(max(1, width - 512 + 1)))
        try:
            crop = np.asarray(array[z, y:y + 512, x:x + 512])
        except Exception as exc:
            raise RuntimeError(f'Image read failed: volume={vid} z={z} y={y} x={x}') from exc
        if crop.ndim != 2 or min(crop.shape) < 1 or not np.isfinite(crop).all():
            raise ValueError(f'Invalid crop: volume={vid} z={z} y={y} x={x} shape={crop.shape}')
        py, px = 512 - crop.shape[0], 512 - crop.shape[1]
        if py or px:
            crop = np.pad(crop, ((0, py), (0, px)),
                          mode='reflect' if min(crop.shape) > 1 else 'edge')
        # PIL's RGB conversion replicates the grayscale channel for the official patch embedding.
        image = Image.fromarray(crop).convert('RGB')
        location = {'index': absolute_index, 'slice_number': slice_number,
                    'source': row['source_dataset'], 'volume_id': vid,
                    'z': z, 'y': y, 'x': x,
                    'view_seed': int(rng.integers(2**31 - 1))}
        return image, location

    def __getitem__(self, index: int):
        local_index = self.start + index
        if self.world_size > 1:
            batch_number, within_batch = divmod(local_index, self.batch_per_rank)
            absolute_index = (batch_number * self.world_size + self.rank) * self.batch_per_rank + within_batch
        else:
            absolute_index = local_index
        image, location = self.raw(absolute_index)
        seed = location['view_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return self.aug(image), location


class Collate:
    def __init__(self, cfg):
        self.cfg = cfg
        token_size = cfg.crops.global_crops_size // cfg.student.patch_size
        self.mask_generator = MaskingGenerator(input_size=(token_size, token_size),
                                               max_num_patches=token_size**2 // 2)
        self.tokens = token_size**2

    def __call__(self, samples):
        locations = [sample[1] for sample in samples]
        first = locations[0]['index']
        seed = 1_700_000_000 + first
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        result = collate_data_and_cast(
            [(sample[0], sample[1]['index']) for sample in samples],
            mask_ratio_tuple=self.cfg.ibot.mask_ratio_min_max,
            mask_probability=self.cfg.ibot.mask_sample_probability,
            dtype=torch.float32,
            n_tokens=self.tokens,
            mask_generator=self.mask_generator,
        )
        result['_first_index'] = first
        result['_sample_locations'] = locations
        result['_sample_digest'] = hashlib.sha256(
            json.dumps(locations, sort_keys=True).encode()).hexdigest()
        return result
