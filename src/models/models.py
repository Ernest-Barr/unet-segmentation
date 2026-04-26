import torch
import torch.nn.functional as F

from .ModularUNet import ModularUNet
from .UNetPP import UNetPPModel
from .layers import DoubleConv, ResidualConv, DenseConv

def UNet(imgChannels=3, outChannels=1, initFeatures=32, depth=4, dropRate=0.1):
    return ModularUNet(
        imgChannels=imgChannels,
        outChannels=outChannels,
        initFeatures=initFeatures,
        depth=depth,
        attention=False,
        conv_block=DoubleConv,
        dropRate=dropRate
    )

def AttentionUNet(imgChannels=3, outChannels=1, initFeatures=32, depth=4, dropRate=0.1):
    return ModularUNet(
        imgChannels=imgChannels,
        outChannels=outChannels,
        initFeatures=initFeatures,
        depth=depth,
        attention=True,
        conv_block=DoubleConv,
        dropRate=dropRate
    )

def ResUNet(imgChannels=3, outChannels=1, initFeatures=32, depth=4, dropRate=0.1):
    return ModularUNet(
        imgChannels=imgChannels,
        outChannels=outChannels,
        initFeatures=initFeatures,
        depth=depth,
        attention=False,
        conv_block=ResidualConv,
        dropRate=dropRate
    )

def ResAttentionUNet(imgChannels=3, outChannels=1, initFeatures=32, depth=4, dropRate=0.1):
    return ModularUNet(
        imgChannels=imgChannels,
        outChannels=outChannels,
        initFeatures=initFeatures,
        depth=depth,
        attention=True,
        conv_block=ResidualConv,
        dropRate=dropRate
    )

def DenseUNet(imgChannels=3, outChannels=1, initFeatures=32, depth=4, dropRate=0.1):
    return ModularUNet(
        imgChannels=imgChannels,
        outChannels=outChannels,
        initFeatures=initFeatures,
        depth=depth,
        attention=False,
        conv_block=DenseConv,
        dropRate=dropRate
    )

def DenseAttentionUNet(imgChannels=3, outChannels=1, initFeatures=32, depth=4, dropRate=0.1):
    return ModularUNet(
        imgChannels=imgChannels,
        outChannels=outChannels,
        initFeatures=initFeatures,
        depth=depth,
        attention=True,
        conv_block=DenseConv,
        dropRate=dropRate
    )

def UNetPP(imgChannels=3, outChannels=1, initFeatures=32, depth=4, dropRate=0.1):
    return UNetPPModel(
        imgChannels=imgChannels,
        outChannels=outChannels,
        initFeatures=initFeatures,
        depth=depth,
        attention=False,
        conv_block=DoubleConv,
        dropRate=dropRate
    )

# https://www.geeksforgeeks.org/python/args-kwargs-python/
def get_model(model_name: str, **kwargs):

    model_dict = {
        'UNet': UNet,
        'AttentionUNet': AttentionUNet,
        'ResUNet': ResUNet,
        'ResAttentionUNet': ResAttentionUNet,
        'DenseUNet': DenseUNet,
        'DenseAttentionUNet': DenseAttentionUNet,
        'UNetPP': UNetPP,
    }

    if model_name in model_dict:
        return model_dict[model_name](**kwargs)

    return ValueError("Invalid model name")