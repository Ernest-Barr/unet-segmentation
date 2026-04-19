# import torch
# from tqdm import tqdm
# from torch.utils.data import DataLoader
#
# class Trainer:
#     def __init__(self,
#                  model,
#                  trainLoader,
#                  valLoader,
#                  optimizer,
#                  criterion,
#                  device):
#         self.model = model.to(device)
#         self.trainLoader = trainLoader
#         self.valLoader = valLoader
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.device = device
#
#     def trainOneEpoch(self):
#         self.model.train()
#         trainRunningLoss = 0.0
#
#         for img, label in tqdm(self.trainLoader):
#             img = img.float().to(self.device)
#             label = label.float().to(self.device)
#
#             self.optimizer.zero_grad()
#             yPred = self.model(img)
#             loss = self.criterion(yPred, label)
#
#             loss.backward()
#             self.optimizer.step()
#
#             trainRunningLoss += loss.item()
#         return trainRunningLoss / len(self.trainLoader)
#
#     def validateOneEpoch(self):
#         self.model.eval()
#         valRunningLoss = 0.0
#         # totalCorrect = 0
#         # totalPixels = 0
#
#         with torch.no_grad():
#             for img, label in tqdm(self.valLoader):
#                 img = img.float().to(self.device)
#                 label = label.float().to(self.device)
#
#                 yPred = self.model(img)
#                 loss = self.criterion(yPred, label)
#
#                 valRunningLoss += loss.item()
#
#                 preds = (torch.sigmoid(yPred) > 0.5).float()
#
#                 # totalCorrect += (preds == label).sum().item()
#                 # totalPixels += label.numel()
#
#         avgLoss = valRunningLoss / len(self.valLoader)
#         # accuracy = totalCorrect / totalPixels
#
#         return avgLoss
#
#     def train(self, numEpochs: int, earlyStopping=True, patience=5, saveModel=False, saveModelPath="trainedModel/temp.pth"):
#         bestValLoss = 99999
#         numNoImprovement = 0
#
#         for e in range(numEpochs):
#             print(f"Epoch {e+1}")
#             trainLoss = self.trainOneEpoch()
#             valLoss = self.validateOneEpoch()
#
#             print(f"Epoch {e+1} Train Loss: {trainLoss:.4f}")
#             print(f"Epoch {e+1} Val Loss:   {valLoss:.4f}")
#
#             if valLoss < bestValLoss:
#                 bestValLoss = valLoss
#                 numNoImprovement = 0
#                 if saveModel:
#                     torch.save(self.model.state_dict(), saveModelPath)
#                     print("Validation improved — model saved.")
#                 else:
#                     print("Validation improved")
#             else:
#                 numNoImprovement += 1
#                 print("No validation improvement")
#
#             if earlyStopping:
#                 if numNoImprovement >= patience:
#                     print("Early stopping triggered.")
#                     break