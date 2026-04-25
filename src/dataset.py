import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import medsegbench
import config

from typing import List, Any
from PIL import Image
from torch.utils.data import Dataset


# https://docs.pytorch.org/vision/0.8/transforms.html
# https://docs.pytorch.org/vision/stable/transforms.html
# https://docs.pytorch.org/vision/0.9/transforms.html
# https://www.geeksforgeeks.org/python/python-unpack-list/

class MedSegBenchDataset(Dataset):
    def __init__(self, dataset_instance, split='train', meta=None):
        self.dataset = dataset_instance
        self.split = split
        self.meta = meta

        img_transforms: List[Any] = [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        ]

        if split == 'train':
            img_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])

        img_transforms.extend([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        self.transform = transforms.Compose(img_transforms)

        mask_base: List[Any] = [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToImage(),
        ]

        if self.meta['task'] == 'binary':
            mask_base.append(transforms.ToDtype(torch.float32, scale=True))
        else:
            mask_base.append(transforms.ToDtype(torch.long, scale=False))

        if split == 'train':
            self.mask_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                *mask_base
            ])
        else:
            self.mask_transform = transforms.Compose(mask_base)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        #    https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image

        state = torch.get_rng_state()
        image = self.transform(image)

        torch.set_rng_state(state)
        mask = self.mask_transform(mask)

        return image, mask


def get_dataset(dataset_name: str, split: str = 'train', download: bool = True, **kwargs):
    if dataset_name not in config.DATASET_METADATA:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    dataset_class = getattr(medsegbench, dataset_name)

    metadata = config.DATASET_METADATA[dataset_name]

    dataset_instance = dataset_class(
        split=split,
        download=download,
        size=config.IMAGE_SIZE,
        root=config.DATA_DIR,
        **kwargs
    )

    return MedSegBenchDataset(dataset_instance, split=split, meta=metadata), metadata
