import torch
import torch.nn as nn
from .layers import DoubleConv, Encoder, Decoder


class ModularUNet(nn.Module):
    def __init__(self, imgChannels: int, outChannels: int, initFeatures: int, depth: int, attention: bool = False,
                 conv_block=DoubleConv, dropRate=0.1):
        """depth: start at 0"""
        super(ModularUNet, self).__init__()

        self.downLayers = nn.ModuleList()
        self.upLayers = nn.ModuleList()

        self.features = [initFeatures * (2 ** (i)) for i in range(depth)]

        for numF in self.features:
            self.downLayers.append(Encoder(imgChannels, numF, conv_block=DoubleConv, p=dropRate))
            imgChannels = numF

        self.bottleneckLayer = DoubleConv(self.features[-1], self.features[-1] * 2, p=dropRate)

        for i in reversed(range(depth)):
            skip_channels = self.features[i]
            in_channels = self.features[i] * 2
            out_channels = self.features[i]

            self.upLayers.append(
                Decoder(in_channels, skip_channels, out_channels, conv_block=DoubleConv, attention=attention,
                        p=dropRate))

        self.finalConv = nn.Conv2d(self.features[0], outChannels, kernel_size=1)

    def forward(self, x):
        skipConnections = []

        for encoder in self.downLayers:
            features, x = encoder(x)
            skipConnections.append(features)

        x = self.bottleneckLayer(x)
        skipConnections.reverse()

        for i, decoder in enumerate(self.upLayers):
            x = decoder(x, skipConnections[i])

        return self.finalConv(x)
