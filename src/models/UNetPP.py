# import torch
# import torch.nn as nn
# from layers import DoubleConv
#
#
# class ModularUNetPP(nn.Module):
#     def __init__(self, imgChannels: int, outChannels: int, initFeatures: int, depth: int, dropRate=0.1):
#         """depth: start at 0"""
#         super(UNetPP, self).__init__()
#
#         self.downLayers = nn.ModuleList()
#         self.features = [initFeatures * (2 ** (i)) for i in range(depth)]
#         self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.upSamples = nn.ModuleDict()
#         self.upConvs = nn.ModuleDict()
#
#         for numF in self.features:
#             self.downLayers.append(DoubleConv(imgChannels, numF))
#             imgChannels = numF
#
#         self.bottleneckLayer = DoubleConv(self.features[-1], self.features[-1] * 2)
#         # Append the number of features in bottleneckLayer
#         self.features.append(self.features[-1] * 2)
#
#         for curCol in range(1, len(self.features)):
#             for curRow in range(len(self.features) - curCol):
#                 inC = self.features[curRow] * curCol + self.features[curRow + 1]
#                 outC = self.features[curRow]
#
#                 self.upSamples[f"{curRow}_{curCol}"] = nn.ConvTranspose2d(
#                     self.features[curRow + 1],
#                     self.features[curRow + 1],
#                     kernel_size=2,
#                     stride=2
#                 )
#
#                 self.upConvs[f"{curRow}_{curCol}"] = DoubleConv(inC, outC)
#
#         self.finalConv = nn.Conv2d(self.features[0], outChannels, kernel_size=1)
#
#         # print(self.downLayers)
#         # print(self.bottleneckLayer)
#         # print(self.upSamples)
#         # print(self.upConvs)
#
#     def forward(self, x):
#         # n x n list
#         skipConnections = []
#         n = len(self.features)
#
#         for curRow in range(n - 1):
#             x = self.downLayers[curRow](x)
#             skipConnections.append([x])
#             x = self.maxPool(x)
#
#         x = self.bottleneckLayer(x)
#         skipConnections.append([x])
#
#         for curCol in range(1, n):
#             for curRow in range(n - curCol):
#                 catList = []
#                 for i in range(curCol):
#                     catList.append(skipConnections[curRow][i])
#
#                 # Up sample the image on the next row and append it to catList
#                 catList.append(self.upSamples[f"{curRow}_{curCol}"](skipConnections[curRow + 1][curCol - 1]))
#                 merged = torch.cat(catList, dim=1)
#                 merged = self.upConvs[f"{curRow}_{curCol}"](merged)
#                 skipConnections[curRow].append(merged)
#
#         return self.finalConv(skipConnections[0][n - 1])