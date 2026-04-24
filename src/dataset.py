import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import medsegbench
from PIL import Image
import config

# https://docs.pytorch.org/vision/0.8/transforms.html
# https://docs.pytorch.org/vision/stable/transforms.html

class MedSegBenchDataset(Dataset):
    def __init__(self, dataset_instance, split='train'):
        self.dataset = dataset_instance
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True)
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True)
            ])

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

        if mask.mode != 'L':
            mask = mask.convert('L')

        image, mask = self.transform(image, mask)

        return image, mask


def get_dataset(dataset_name: str, split: str = 'train', download: bool = True, **kwargs):

    dataset_dict = {
        'AbdomenUSMSBench': medsegbench.AbdomenUSMSBench,
        'Bbbc010MSBench': medsegbench.Bbbc010MSBench,
        'Bkai-Igh-MSBench': medsegbench.BkaiIghMSBench,
        'BriFiSegMSBench': medsegbench.BriFiSegMSBench,
        'BusiMSBench': medsegbench.BusiMSBench,
        'CellnucleiMSBench': medsegbench.CellnucleiMSBench,
        'ChaseDB1MSBench': medsegbench.ChaseDB1MSBench,
        'ChuacMSBench': medsegbench.ChuacMSBench,
        'Covid19RadioMSBench': medsegbench.Covid19RadioMSBench,
        'CovidQUExMSBench': medsegbench.CovidQUExMSBench,
        'CystoFluidMSBench': medsegbench.CystoFluidMSBench,
        'Dca1MSBench': medsegbench.Dca1MSBench,
        'DeepbacsMSBench': medsegbench.DeepbacsMSBench,
        'DriveMSBench': medsegbench.DriveMSBench,
        'DynamicNuclearMSBench': medsegbench.DynamicNuclearMSBench,
        'FHPsAOPMSBench': medsegbench.FHPsAOPMSBench,
        'IdribMSBench': medsegbench.IdribMSBench,
        'Isic2016MSBench': medsegbench.Isic2016MSBench,
        'Isic2018MSBench': medsegbench.Isic2018MSBench,
        'KvasirMSBench': medsegbench.KvasirMSBench,
        'M2caiSegMSBench': medsegbench.M2caiSegMSBench,
        'MonusacMSBench': medsegbench.MonusacMSBench,
        'MosMedPlusMSBench': medsegbench.MosMedPlusMSBench,
        'NucleiMSBench': medsegbench.NucleiMSBench,
        'NusetMSBench': medsegbench.NusetMSBench,
        'PandentalMSBench': medsegbench.PandentalMSBench,
        'PolypGenMSBench': medsegbench.PolypGenMSBench,
        'Promise12MSBench': medsegbench.Promise12MSBench,
        'RoboToolMSBench': medsegbench.RoboToolMSBench,
        'TnbcnucleiMSBench': medsegbench.TnbcnucleiMSBench,
        'UltrasoundNerveMSBench': medsegbench.UltrasoundNerveMSBench,
        'USforKidneyMSBench': medsegbench.USforKidneyMSBench,
        'UWSkinCancerMSBench': medsegbench.UWSkinCancerMSBench,
        'WbcMSBench': medsegbench.WbcMSBench,
        'YeazMSBench': medsegbench.YeazMSBench,
    }

    if dataset_name not in dataset_dict:
        raise ValueError(f"Invalid dataset: {dataset_name}. Choose from {list(dataset_dict.keys())}")

    dataset_instance = dataset_dict[dataset_name](
        split=split,
        download=download,
        size=config.IMAGE_SIZE,
        root=config.DATA_DIR,
        **kwargs
    )

    return MedSegBenchDataset(dataset_instance, split=split)


