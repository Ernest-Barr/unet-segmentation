import torch
import torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from trainer import Trainer
from dataset import getDataset
from model import UNet, UNetPP
import constants as const

savePath = "trainedModel/UNet1.pth"
savePathPP = "trainedModel/UNetPP1.pth"
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

trainDataset, valDataset, testDataset = getDataset(const.CELLNUCLEI_FILE_PATH)

model = UNet(const.CELLNUCLEI_IN_CHANNELS, const.CELLNUCLEI_OUT_CHANNELS, 32, 4)
modelPP = UNetPP(const.CELLNUCLEI_IN_CHANNELS, const.CELLNUCLEI_OUT_CHANNELS, 32, 4)

trainLoader = DataLoader(dataset=trainDataset, batch_size=4, shuffle=True)
valLoader = DataLoader(dataset=valDataset, batch_size=4, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

trainer = Trainer(modelPP, trainLoader, valLoader, optimizer, criterion, device)
trainer.train(100, patience=10, saveModel=True, saveModelPath=savePath)


# model = model.to(device)
# modelPP = modelPP.to(device)
# model.load_state_dict(torch.load(savePath, map_location=device))
# modelPP.load_state_dict(torch.load(savePathPP, map_location=device))

# model.eval()
# modelPP.eval()

# for i in range (20, 30):
#     img, mask = testDataset[i]

#     with torch.no_grad():
#         img_tensor = img.unsqueeze(0).to(device)

#         # UNet prediction
#         pred_unet = model(img_tensor)
#         pred_unet = torch.sigmoid(pred_unet)
#         mask_unet = (pred_unet > 0.5).float().squeeze().cpu().numpy()

#         # UNetPP prediction
#         pred_unetpp = modelPP(img_tensor)
#         pred_unetpp = torch.sigmoid(pred_unetpp)
#         mask_unetpp = (pred_unetpp > 0.5).float().squeeze().cpu().numpy()

#     plt.figure(figsize=(20, 5))

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

#     # UNetPP prediction
#     plt.subplot(1, 4, 3)
#     plt.imshow(mask_unetpp, cmap="gray")
#     plt.title("UNetPP Prediction")
#     plt.axis("off")

#     # Ground truth
#     plt.subplot(1, 4, 4)
#     plt.imshow(mask.squeeze(), cmap="gray")
#     plt.title("Ground Truth")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()