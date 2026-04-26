import torch
import torch.nn as nn
from .layers import Encoder, DoubleConv


class UNetPPModel(nn.Module):
    def __init__(self, imgChannels: int, outChannels: int, initFeatures: int, depth: int, attention=False, 
                 conv_block=DoubleConv, dropRate=0.1):
        """depth: start at 0"""
        super(UNetPPModel, self).__init__()

        self.downLayers = nn.ModuleList()
        self.features = [initFeatures * (2 ** (i)) for i in range(depth)]

        self.upSamples = nn.ModuleDict()
        self.upConvs = nn.ModuleDict()

        for numF in self.features:
            self.downLayers.append(Encoder(imgChannels, numF, conv_block, dropRate))
            imgChannels = numF

        self.bottleneckLayer = DoubleConv(self.features[-1], self.features[-1] * 2, dropRate)
        # Append the number of features in bottleneckLayer
        self.features.append(self.features[-1] * 2)

        for curCol in range(1, len(self.features)):
            for curRow in range(len(self.features) - curCol):
                inC = self.features[curRow] * curCol + self.features[curRow + 1]
                outC = self.features[curRow]

                self.upSamples[f"{curRow}_{curCol}"] = nn.ConvTranspose2d(
                    self.features[curRow + 1],
                    self.features[curRow + 1],
                    kernel_size=2,
                    stride=2
                )

                self.upConvs[f"{curRow}_{curCol}"] = DoubleConv(inC, outC)

        self.finalConv = nn.Conv2d(self.features[0], outChannels, kernel_size=1)

    def forward(self, x):
        # n x n list
        skipConnections = []
        n = len(self.features)

        for curRow in range(n - 1):
            f, x = self.downLayers[curRow](x)
            skipConnections.append([f])

        x = self.bottleneckLayer(x)
        skipConnections.append([x])

        for curCol in range(1, n):
            for curRow in range(n - curCol):
                catList = []
                for i in range(curCol):
                    catList.append(skipConnections[curRow][i])

                # Up sample the image on the next row and append it to catList
                catList.append(self.upSamples[f"{curRow}_{curCol}"](skipConnections[curRow + 1][curCol - 1]))
                merged = torch.cat(catList, dim=1)
                merged = self.upConvs[f"{curRow}_{curCol}"](merged)
                skipConnections[curRow].append(merged)

        return self.finalConv(skipConnections[0][n - 1])