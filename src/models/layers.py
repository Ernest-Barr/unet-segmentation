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


class ResidualConv(nn.Module):
    def __init__(self, inChannels, outChannels, p=0.1):
        super(ResidualConv, self).__init__()
        self.conv = DoubleConv(inChannels, outChannels, p)
        self.shortcut = nn.Conv2d(inChannels, outChannels, 1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class AttentionGate(nn.Module):
    def __init__(self, g, x, out):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(g, out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x, out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=DoubleConv, p=0.1):
        super(Encoder, self).__init__()
        self.conv = conv_block(in_channels, out_channels, p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        down = self.pool(features)

        return features, down


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, conv_block=DoubleConv, attention=False, p=0.1):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention = attention

        if self.attention:
            self.att_gate = AttentionGate(g=in_channels // 2, x=skip_channels, out=in_channels // 2)

        self.conv = conv_block((in_channels // 2) + skip_channels, out_channels, p)

    def forward(self, x, skip_connections):
        x = self.up(x)

        if isinstance(skip_connections, (list, tuple)):
            skip_features = torch.cat(skip_connections, dim=1)
        else:
            skip_features = skip_connections

        if self.attention:
            skip_features = self.att_gate(g=x, x=skip_features)

        concat_x = torch.cat([skip_features, x], dim=1)

        return self.conv(concat_x)
