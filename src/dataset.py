import numpy as np
import torch
from torch.utils.data import Dataset


def parseNpData(data, imgNumDim: int, labelNumDim: int):
    """Return (trainImages, trainLabels, valImages, valLabels, testImages, testLabels)"""
    # np dim format is <H, W, C>, tensor is <C, H, W>
    # If the dim is 4, reorder to match tensor format, if dim is 3, add extra dim for channel
    trainImages = data["train_images"].transpose(0, 3, 1, 2) if imgNumDim == 4 else np.expand_dims(data["train_images"], axis=1)
    valImages = data["val_images"].transpose(0, 3, 1, 2) if imgNumDim == 4 else np.expand_dims(data["val_images"], axis=1)
    testImages = data["test_images"].transpose(0, 3, 1, 2) if imgNumDim == 4 else np.expand_dims(data["test_images"], axis=1)

    trainLabels = data["train_label"].transpose(0, 3, 1, 2) if labelNumDim == 4 else np.expand_dims(data["train_label"], axis=1)
    valLabels = data["val_label"].transpose(0, 3, 1, 2) if labelNumDim == 4 else np.expand_dims(data["val_label"], axis=1)
    testLabels = data["test_label"].transpose(0, 3, 1, 2) if labelNumDim == 4 else np.expand_dims(data["test_label"], axis=1)

    return trainImages, trainLabels, valImages, valLabels, testImages, testLabels

def getDataset(dataFilePath: str, transform = None):
    """Return (trainImageDataset, valImageDataset, testImageDataset)"""
    data = np.load(dataFilePath)
    trainImages, trainLabels, valImages, valLabels, testImages, testLabels = parseNpData(data, data["train_images"].ndim, data["train_label"].ndim)
    return (ModelDataset(trainImages, trainLabels, transform), 
            ModelDataset(valImages, valLabels, transform), 
            ModelDataset(testImages, testLabels, transform))


class ModelDataset(Dataset):
    def __init__(self, images, labels, transform = None):   
        self.images = images
        self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].astype(np.float32)
        label = self.labels[index].astype(np.float32)
        
        image = torch.from_numpy(image) / 255.0
        label = torch.from_numpy(label)

        # Note: transform is not used on label
        if self.transform:
            return self.transform(image), label
        else:
            return image, label
        
# train, val, test = getDataset(const.COVID_FILE_PATH)

# img, mask = train[1]

# print(f"Image shape: {img.shape}")  # should be (1, 256, 256)
# print(f"Mask shape: {mask.shape}")  # should be (1, 256, 256)
# print(f"Image min/max: {img.min().item()}/{img.max().item()}")
# print(f"Mask min/max: {mask.min().item()}/{mask.max().item()}")