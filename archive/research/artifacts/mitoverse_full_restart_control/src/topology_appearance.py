"""Appearance candidates shared by offline PH calibration and SSL training."""

import numpy as np
from scipy.ndimage import gaussian_filter

CANDIDATES = [
    {"name": "base", "gamma": [0.8, 1.2], "contrast": [0.8, 1.2], "noise": 0.01, "blur": 0.3},
    {"name": "gamma_mild", "gamma": [0.8, 1.2]},
    {"name": "gamma_broad", "gamma": [0.6, 1.4]},
    {"name": "contrast_mild", "contrast": [0.8, 1.2]},
    {"name": "noise_001", "noise": 0.01},
    {"name": "noise_002", "noise": 0.02},
    {"name": "noise_003", "noise": 0.03},
    {"name": "blur_03", "blur": 0.3},
    {"name": "blur_06", "blur": 0.6},
    {"name": "blur_10", "blur": 1.0},
    {"name": "gamma_noise", "gamma": [0.8, 1.2], "noise": 0.02},
    {"name": "contrast_blur", "contrast": [0.8, 1.2], "blur": 0.6},
]


def appearance(image: np.ndarray, candidate: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32, copy=True)
    if "gamma" in candidate:
        low, high = candidate["gamma"]
        x = np.power(np.clip(x, 0, 1), float(rng.uniform(low, high)))
    if "contrast" in candidate:
        low, high = candidate["contrast"]
        factor = float(rng.uniform(low, high))
        x = (x - float(x.mean())) * factor + float(x.mean())
    if "blur" in candidate:
        x = gaussian_filter(x, sigma=float(candidate["blur"]), mode="reflect")
    if "noise" in candidate:
        x = x + rng.normal(0.0, float(candidate["noise"]), size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32)
