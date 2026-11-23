from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from heads_model import CONFIG, PREVIOUS
from model import BASE, read, sha
from batch4_prepare import CACHE4
from bench_inputs import cem_read
from mixture import C
from heads_targets import instance_target


class CEMDraws:
    def __init__(self):
        self.raw, self.ids, flags = (np.load(CACHE4/"cem"/p, mmap_mode="r")
                                    for p in ("raw.npy", "labels.npy", "instance.npy"))
        assert self.raw.shape == self.ids.shape == (160000,256,256) and flags.all()
        token = BASE/"batch4/cem_draws.jsonl"
        self.prefix_sha = sha(token)
        assert self.prefix_sha == read(BASE/"batch4/ready.json")["draw_sha"]["cem"]
        self.groups = read(C/"manifest.json")
        self.collections = sorted(set(self.groups["groups"])-set(self.groups["validation"]))
        self.image_count = sum(len(self.groups["groups"][g]) for g in self.collections)

    @staticmethod
    @lru_cache(maxsize=12000)
    def full(g, n):
        return cem_read(g,n)

    def generate(self, i):
        rng = np.random.default_rng(np.random.SeedSequence([CONFIG["seed"],i,1]))
        g = self.collections[rng.integers(len(self.collections))]
        names = self.groups["groups"][g]
        name = names[rng.integers(len(names))][0]
        raw, ids = self.full(g,name)
        h, w = raw.shape
        y, x = int(rng.integers(max(1,h-255))), int(rng.integers(max(1,w-255)))
        raw, ids = raw[y:y+256,x:x+256], ids[y:y+256,x:x+256]
        pads = tuple((0,256-v) for v in raw.shape)
        raw = np.pad(raw,pads,constant_values=127)
        values, inv = np.unique(ids,return_inverse=True)
        ids = np.pad((inv.reshape(ids.shape)+int(values[0]!=0)).astype(np.int32),
                     pads,constant_values=-1)
        for axis in (0,1):
            if rng.integers(2):
                raw, ids = np.flip(raw,axis), np.flip(ids,axis)
        return raw, ids

    def verify_extended_sampler(self):
        for i in (6,54,108,111,112,159999):
            raw, ids = self.generate(i)
            assert np.array_equal(self.raw[i],raw) and np.array_equal(self.ids[i],ids), i

    def sample(self, arm, i):
        raw, ids = (self.raw[i],self.ids[i]) if i < len(self.raw) else self.generate(i)
        return raw, ids, instance_target(arm,ids)

    def batches(self, arm, first, last, batch, exposure_offset=0):
        with ThreadPoolExecutor(max_workers=8 if arm != "batch32" else 24) as pool:
            def request(step):
                s = step*batch+exposure_offset
                return [pool.submit(self.sample,arm,i) for i in range(s,s+batch)]

            ahead = {}
            for i in range(first,min(first+4,last)):
                ahead[i] = request(i)
            for i in range(first,last):
                if i+4 < last:
                    ahead[i+4] = request(i+4)
                yield tuple(np.stack(chunk) for chunk in zip(*(f.result() for f in ahead.pop(i))))
