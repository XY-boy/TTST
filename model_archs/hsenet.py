from model_archs import common

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

def make_model(args, parent=False):
    return HSENET(args)


# ref:NONLocalBlock2D
class AdjustedNonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(AdjustedNonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1):

        batch_size = x0.size(0)

        g_x = self.g(x0).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x1).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x0).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # use embedding gaussian
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x0.size()[2:])
        W_y = self.W(y)
        z = W_y + x0

        return z


# hybrid-scale self-similarity exploitation module
class HSEM(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(HSEM, self).__init__()

        base_scale = []
        base_scale.append(SSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        down_scale = []
        down_scale.append(SSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        tail = []
        tail.append(common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        self.NonLocal_base = AdjustedNonLocalBlock(n_feats, n_feats // 2)

        self.base_scale = nn.Sequential(*base_scale)
        self.down_scale = nn.Sequential(*down_scale)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        add_out = x

        # base scale
        x_base = self.base_scale(x)

        # 1/2 scale
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_down = self.down_scale(x_down)

        # fusion x_down and x_down2
        x_down = F.interpolate(x_down, size=(x_base.shape[2], x_base.shape[3]),
                               mode='bilinear')
        ms = self.NonLocal_base(x_base, x_down)
        ms = self.tail(ms)

        add_out = add_out + ms

        return add_out


# single-scale self-similarity exploitation module
class SSEM(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(SSEM, self).__init__()

        head = []
        head.append(common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        MB = [] # main branch
        MB.append(common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))
        MB.append(common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        AB = []  # attention branch
        AB.append(common.NonLocalBlock2D(n_feats, n_feats//2))
        AB.append(nn.Conv2d(n_feats, n_feats, 1, padding=0, bias=True))

        sigmoid = []
        sigmoid.append(nn.Sigmoid())

        tail = []
        tail.append(common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        self.head = nn.Sequential(*head)
        self.MB = nn.Sequential(*MB)
        self.AB = nn.Sequential(*AB)
        self.sigmoid = nn.Sequential(*sigmoid)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        add_out = x
        x_head = self.head(x)
        x_MB = self.MB(x_head)
        x_AB = self.AB(x_head)
        x_AB = self.sigmoid(x_AB)
        x_MB_AB = x_MB * x_AB
        x_tail = self.tail(x_MB_AB)

        add_out = add_out + x_tail
        return add_out


# multi-scale self-similarity block
class BasicModule(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(BasicModule, self).__init__()

        head = [
            common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act)
            for _ in range(2)
        ]

        body = []
        body.append(HSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        tail = [
            common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act)
            for _ in range(2)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        add_out = x

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        add_out = add_out + x

        return add_out


class HSENET(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(HSENET, self).__init__()

        n_feats = 128
        kernel_size = 3
        scale = 4
        act = nn.ReLU(True)

        self.n_BMs = 10

        # rgb_mean = (0.4916, 0.4991, 0.4565)  # UCMerced data
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head body
        m_head = [conv(3, n_feats, kernel_size)]

        # define main body
        self.body_modulist = nn.ModuleList([
            BasicModule(conv, n_feats, kernel_size, act=act)
            for _ in range(self.n_BMs)
        ])

        # define tail body
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        # main body
        add_out = x

        for i in range(self.n_BMs):
            x = self.body_modulist[i](x)
        add_out = add_out + x

        x = self.tail(add_out)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

if __name__ == '__main__':
    input = torch.rand(1, 3, 128, 128).cuda()  # B C H W
    model = HSENET().cuda()
    flops, params = profile(model, inputs=(input,))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))

    # output = model(input)
    # print(output.size())
