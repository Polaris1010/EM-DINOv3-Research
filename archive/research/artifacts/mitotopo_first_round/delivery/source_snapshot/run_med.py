# -*- coding: utf-8 -*-
import sys
import torchvision
import dinov3.data.loaders as loaders
from dinov3.train.train import main

class CustomImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, transforms=None, **kwargs):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

original_parse = loaders._parse_dataset_str

def custom_parse(dataset_str):
    if dataset_str.startswith("ImageFolder"):
        root_dir = dataset_str.split("root=")[-1]
        return CustomImageFolder, {"root": root_dir}
    return original_parse(dataset_str)

loaders._parse_dataset_str = custom_parse

if __name__ == "__main__":
    print("[INFO] Launching DINOv3 with Custom ImageFolder Support (Continual Pretraining)!")
    sys.exit(main())