import torch
import torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from trainer import Trainer
from dataset import getDataset
from model import UNet
import constants as const

savePath = "trainedModel/UNet1.pth"
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

trainDataset, valDataset, testDataset = getDataset(const.CELLNUCLEI_FILE_PATH)

model = UNet(const.CELLNUCLEI_IN_CHANNELS, const.CELLNUCLEI_OUT_CHANNELS, 32, 4, dropRate=0.2)

trainLoader = DataLoader(dataset=trainDataset, batch_size=4, shuffle=True)
valLoader = DataLoader(dataset=valDataset, batch_size=4, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

trainer = Trainer(model, trainLoader, valLoader, optimizer, criterion, device)
trainer.train(50, patience=5, saveModel=True, saveModelPath=savePath)


# model = model.to(device)
# model.load_state_dict(torch.load(savePath, map_location=device))

# model.eval()

# for i in range (5):
#     img, mask = testDataset[i]

#     with torch.no_grad():
#         img_tensor = img.unsqueeze(0).to(device)

#         # UNet prediction
#         pred_unet = model(img_tensor)
#         pred_unet = torch.sigmoid(pred_unet)
#         mask_unet = (pred_unet > 0.5).float().squeeze().cpu().numpy()

#     plt.figure(figsize=(15, 5))

#     # Original image
#     plt.subplot(1, 4, 1)
#     plt.imshow(img.permute(1, 2, 0))
#     plt.title("Original Image")
#     plt.axis("off")

#     # UNet prediction
#     plt.subplot(1, 4, 2)
#     plt.imshow(mask_unet, cmap="gray")
#     plt.title("UNet Prediction")
#     plt.axis("off")

#     # Ground truth
#     plt.subplot(1, 4, 3)
#     plt.imshow(mask.squeeze(), cmap="gray")
#     plt.title("Ground Truth")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()