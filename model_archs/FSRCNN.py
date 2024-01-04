import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
import math
import pdb
import time
import numpy as np
from math import sqrt
import argparse
from thop import profile


class Net(torch.nn.Module):
    def __init__(self, scale=4, n_channels = 3, d=56, s=12, m=4):
        # too big network may leads to over-fitting
        super(Net, self).__init__()

        # Feature extraction
        self.first_part = nn.Sequential(nn.Conv2d(in_channels=n_channels, out_channels=d, kernel_size=3, stride=1, padding=0),
                                        nn.PReLU())
        # H_out = floor((H_in+2*padding-(kernal_size-1)-1)/stride+1)
        #       = floor(H_in-4)
        # for x2  floor(H_in-2)
        self.layers = []
        # Shrinking
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        # Non-linear Mapping
        for _ in range(m):
            self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1),
                                         nn.PReLU()))

        # # Expanding
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=n_channels, kernel_size=9, stride=scale, padding=4, output_padding=0)
        # self.last_part = nn.Sequential(nn.Conv2d(in_channels=d, out_channels=n_channels * 2 * 2, kernel_size=3, stride=1, padding=1),
        #                                nn.PixelShuffle(2))
        # H_out = (H_in-1)*stride-2*padding+kernal_size+out_padding
        #       = (H_in-1)*3+1
        #test input should be (y-5)*3+1
        # for x2 2x-3
        # for x4 4x-25

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self):
        '''
        Initial the weights.
        :return:
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0.0, 0.2)
                m.weight.data.normal_(0.0, sqrt(2/m.out_channels/m.kernel_size[0]/m.kernel_size[0])) # MSRA
                # nn.init.xavier_normal(m.weight) # Xavier
                if m.bias is not None:
                    m.bias.data.zero_()
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))

def runing_time(net, x):
    net = net.cuda()
    
    x = Variable(x.cuda())
    y = net(x)
    print(y.size())
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y = net(x)
        
        timer.toc()

    print('Do once forward need {:.3f}ms '.format(timer.total_time*1000/100.0))
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument("--scale", type=int, default=2, help="scale size")

if __name__ == '__main__':
    input = torch.rand(1, 3, 128, 128).cuda()  # B C H W
    model = Net().cuda()
    flops, params = profile(model, inputs=(input,))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))

    output = model(input)
    print(output.size())
    
    

