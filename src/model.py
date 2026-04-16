import torch
import torch.nn as nn

def DoubleConv(inChannels, outChannels, p=0.1):
    return nn.Sequential(
        nn.Conv2d(inChannels, outChannels, 3, padding="same", bias=False),
        nn.BatchNorm2d(outChannels),
        nn.ReLU(inplace=False),

        nn.Conv2d(outChannels, outChannels, 3, padding="same", bias=False),
        nn.BatchNorm2d(outChannels),
        nn.ReLU(inplace=False),
        nn.Dropout2d(p),
    )

class UNet(nn.Module):
    def __init__(self, imgChannels: int, outChannels: int, initFeatures: int, depth: int, dropRate=0.1):
        """depth: start at 0"""
        super(UNet, self).__init__()

        self.downLayers = nn.ModuleList()
        self.upLayers = nn.ModuleList()
        self.features = [initFeatures * (2**(i)) for i in range(depth)]
    
        for numF in self.features:
            self.downLayers.append(DoubleConv(imgChannels, numF, dropRate))
            self.downLayers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            imgChannels = numF

        self.bottleneckLayer = DoubleConv(self.features[-1], self.features[-1]*2)

        for numF in reversed(self.features):
            self.upLayers.append(nn.ConvTranspose2d(numF*2, numF, kernel_size=2, stride=2))
            self.upLayers.append(DoubleConv(numF*2, numF, dropRate))
        
        self.finalConv = nn.Conv2d(self.features[0], outChannels, kernel_size=1)

    def forward(self, x):
        skipConnections = []

        for i in range(0, len(self.downLayers), 2):
            x = self.downLayers[i](x)
            skipConnections.append(x)
            x = self.downLayers[i+1](x)
        
        x = self.bottleneckLayer(x)
        skipConnections.reverse()
        
        for i in range(0, len(self.downLayers), 2):
            x = self.upLayers[i](x)
            sc = skipConnections[i//2]
            addedSC = torch.cat((sc, x), dim=1)
            x = self.upLayers[i+1](addedSC)
        
        return self.finalConv(x)