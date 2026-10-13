"""Deterministic, image-only, source-balanced 2D Zarr sampling for SSL."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import Dataset

from dinov3.data import DataAugmentationDINO
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.masking import MaskingGenerator


class Images(Dataset):
    def __init__(self, manifest: Path, cfg, seed: int, start: int, count: int, arm: str = 'B'):
        self.start = start
        self.count = count
        self.seed = seed
        self.arm = arm
        self.handles = {}
        self.topology = arm in {'T', 'LT'}
        if self.topology:
            from topology_appearance import CANDIDATES, appearance
            self.appearance = appearance
            self.candidates = {item['name']: item for item in CANDIDATES}
            self.selection = json.loads((manifest.parent.parent / 'topology_calibration/topology_selected_locked.json').read_text())
            if self.selection['status'] != 'complete':
                raise ValueError('Topology candidate selection is not complete')
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            if self.selection.get('manifest_sha256') != digest:
                raise ValueError('Topology calibration uses another manifest')
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        if not rows or any(row.get('training_role') != 'ssl_full_mitoverse' for row in rows):
            raise ValueError('SSL image manifest must contain full MitoVerse volumes only')
        if len({row['volume_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate volume IDs in SSL image manifest')
        groups = defaultdict(list)
        for row in rows:
            groups[row["original_dataset_id"]].append(row)
        self.sources = sorted(groups)
        self.groups = groups
        self.aug = DataAugmentationDINO(
            global_crops_scale=cfg.crops.global_crops_scale,
            local_crops_scale=cfg.crops.local_crops_scale,
            local_crops_number=cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=None,
            gram_teacher_no_distortions=False,
            horizontal_flips=cfg.crops.horizontal_flips,
            teacher_no_color_jitter=self.topology,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
        )

    def __len__(self) -> int:
        return self.count

    def raw(self, absolute_index: int):
        rng = np.random.default_rng(self.seed * 1_000_000_007 + absolute_index)
        source = self.sources[int(rng.integers(len(self.sources)))]
        row = self.groups[source][int(rng.integers(len(self.groups[source])))]
        vid = row["volume_id"]
        if vid not in self.handles:
            if len(self.handles) >= 2:
                self.handles.clear()
            self.handles[vid] = zarr.open_group(row["local_path"], mode="r")["img"]
        array = self.handles[vid]
        if np.dtype(array.dtype) != np.dtype('uint8'):
            raise ValueError(f'SSL image must be uint8 for fixed /255 preprocessing: {vid}')
        depth, height, width = array.shape
        z = int(rng.integers(depth))
        y = int(rng.integers(max(1, height - 512 + 1)))
        x = int(rng.integers(max(1, width - 512 + 1)))
        crop = np.asarray(array[z, y:y + 512, x:x + 512])
        py, px = 512 - crop.shape[0], 512 - crop.shape[1]
        if py or px:
            crop = np.pad(crop, ((0, py), (0, px)),
                          mode='reflect' if min(crop.shape) > 1 else 'edge')
        image = Image.fromarray(crop).convert("RGB")
        view_seed = int(rng.integers(2**31 - 1))
        location = {"index": absolute_index, "source": source, "volume_id": vid,
                    "z": z, "y": y, "x": x, "valid_h": min(512, height - y),
                    "valid_w": min(512, width - x), "view_seed": view_seed}
        return image, location

    def __getitem__(self, index: int):
        image, location = self.raw(self.start + index)
        seed = location["view_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        views = self.aug(image)
        if self.topology:
            rng = np.random.default_rng(seed + 400_000_000)
            def modified(tensor, tier):
                names = self.selection[tier]
                name = names[int(rng.integers(len(names)))]
                candidate = self.candidates[name]
                gray = (tensor[0].float().numpy() * 0.229 + 0.485).clip(0, 1)
                augmented = self.appearance(gray, candidate, int(rng.integers(2**31 - 1)))
                out = torch.from_numpy(augmented)
                return torch.stack([(out - mean) / std for mean, std in
                                    zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            views['global_crops_teacher'] = [modified(t, 'weak') for t in views['global_crops_teacher']]
            views['global_crops'] = [modified(views['global_crops'][0], 'weak'),
                                     modified(views['global_crops'][1], 'strong')]
            tiers = ['weak', 'strong']
            views['local_crops'] = [modified(t, tiers[int(rng.integers(2))]) for t in views['local_crops']]
        return views, location


class Collate:
    def __init__(self, cfg, arm: str = 'B'):
        self.cfg = cfg
        self.arm = arm
        size = cfg.crops.global_crops_size // cfg.student.patch_size
        self.token_size = size

    def __call__(self, samples):
        locations = [sample[1] for sample in samples]
        first = locations[0]["index"]
        seed = 1_700_000_000 + first
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        result = collate_data_and_cast(
            [(sample[0], sample[1]["index"]) for sample in samples],
            mask_ratio_tuple=self.cfg.ibot.mask_ratio_min_max,
            mask_probability=self.cfg.ibot.mask_sample_probability,
            dtype=torch.float32,
            n_tokens=self.token_size**2,
            mask_generator=MaskingGenerator(input_size=(self.token_size, self.token_size),
                                            max_num_patches=self.token_size**2 // 2),
        )
        if self.arm in {'T', 'LT'}:
            result['collated_teacher_global_crops'] = torch.stack(
                [sample[0]['global_crops_teacher'][i]
                 for i in range(2) for sample in samples]
            )
        digest = hashlib.sha256(json.dumps(locations, sort_keys=True).encode()).hexdigest()
        result["_first_index"] = first
        result["_sample_locations"] = locations
        result["_sample_digest"] = digest
        return result
