import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
# import fjn_util
import os

import warnings
warnings.filterwarnings('ignore')
from mmcv.cnn import ConvModule

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        pass

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))

class Default_Conv(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=(3, 3), stride=1, padding=(1, 1), bias=False, groups=1):
        super(Default_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, stride=stride,padding=padding, bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class ConvUpsampler(nn.Sequential):
    def __init__(self, ch_in, ch_out, bias=False, activation=nn.ReLU()):
        super(ConvUpsampler, self).__init__()
        self.conv1 = Default_Conv(ch_in=ch_in, ch_out=ch_out * 4, k_size=3, bias=bias)
        self.ps2 = nn.PixelShuffle(2)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.ps2(x)
        x = self.activation(x)
        return x

class involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size=3,
                 stride=1):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out
