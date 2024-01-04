import torch
import torch.nn as nn

from .comm import defaultConv


class ResidualUnit(nn.Module):
    def __init__(self, inChannel, outChannel, reScale, kernelSize=1, bias=True):
        super().__init__()

        self.reduction = defaultConv(
            inChannel, outChannel//2, kernelSize, bias)
        self.expansion = defaultConv(
            outChannel//2, inChannel, kernelSize, bias)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        res = self.reduction(x)
        res = self.lamRes * self.expansion(res)
        x = self.lamX * x + res

        return x


class ARFB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.RU1 = ResidualUnit(inChannel, outChannel, reScale)
        self.RU2 = ResidualUnit(inChannel, outChannel, reScale)
        self.conv1 = defaultConv(2*inChannel, 2*outChannel, kernelSize=1)
        self.conv3 = defaultConv(2*inChannel, outChannel, kernelSize=3)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):

        x_ru1 = self.RU1(x)
        x_ru2 = self.RU2(x_ru1)
        x_ru = torch.cat((x_ru1, x_ru2), 1)
        x_ru = self.conv1(x_ru)
        x_ru = self.conv3(x_ru)
        x_ru = self.lamRes * x_ru
        x = x*self.lamX + x_ru
        return x
