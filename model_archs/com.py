import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            act = nn.PReLU(num_parameters=out_channels)
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EdgeConv(nn.Module):
    def __init__(self, conv_edge, in_channels, out_channels):
        super(EdgeConv, self).__init__()

        self.conv_edge = conv_edge
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.conv_edge == 'conv1-sobelx':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            # init scale & bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(bias)
            # init template
            self.template = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.template[i, 0, 0, 0] = 1.0
                self.template[i, 0, 1, 0] = 2.0
                self.template[i, 0, 2, 0] = 1.0
                self.template[i, 0, 0, 2] = -1.0
                self.template[i, 0, 1, 2] = -2.0
                self.template[i, 0, 2, 2] = -1.0
            self.template = nn.Parameter(data=self.template, requires_grad=False)

        elif self.conv_edge == 'conv1-sobely':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            # init scale & bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init template
            self.template = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.template[i, 0, 0, 0] = 1.0
                self.template[i, 0, 0, 1] = 2.0
                self.template[i, 0, 0, 2] = 1.0
                self.template[i, 0, 2, 0] = -1.0
                self.template[i, 0, 2, 1] = -2.0
                self.template[i, 0, 2, 2] = -1.0
            self.template = nn.Parameter(data=self.template, requires_grad=False)

        elif self.conv_edge == 'conv1-laplacian':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            # init scale & bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init template
            self.template = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.template[i, 0, 0, 1] = 1.0
                self.template[i, 0, 1, 0] = 1.0
                self.template[i, 0, 1, 2] = 1.0
                self.template[i, 0, 2, 1] = 1.0
                self.template[i, 0, 1, 1] = -4.0
            self.template = nn.Parameter(data=self.template, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        y1 = F.conv2d(input=y0, weight=self.scale * self.template, bias=self.bias, stride=1, groups=self.out_channels)
        return y1


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-Act-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64, at='prelu'):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.act = action_function(nf, at)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return identity + out


def action_function(n_feat, act_type):
    if act_type == 'prelu':
        act = nn.PReLU(num_parameters=n_feat)
    elif act_type == 'relu':
        act = nn.ReLU(inplace=True)
    elif act_type == 'rrelu':
        act = nn.RReLU(lower=-0.05, upper=0.05)
    elif act_type == 'lrelu':
        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'softplus':
        act = nn.Softplus()
    elif act_type == 'linear':
        pass
    else:
        raise ValueError('The type of activation if not support!')
    return act
