import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from model_archs.arfb import ARFB
from model_archs.hfm  import HFM
from model_archs.comm import defaultConv,SELayer

class HPB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.hfm = HFM()
        self.arfb1 = ARFB(inChannel, outChannel, reScale)
        self.arfb2 = ARFB(inChannel, outChannel, reScale)
        self.arfb3 = ARFB(inChannel, outChannel, reScale)
        self.arfbShare = ARFB(inChannel, outChannel, reScale)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.se = SELayer(inChannel)
        self.conv1 = defaultConv(2*inChannel, outChannel, kernelSize=1 )
    def forward(self,x):
        ori = x
        x = self.arfb1(x)
        x = self.hfm(x)
        x = self.arfb2(x)
        x_share = F.interpolate(x,scale_factor=0.5)
        for _ in range(5):
            x_share = self.arfbShare(x_share)
        x_share = self.upsample(x_share)

        x = torch.cat((x_share,x),1)
        x = self.conv1(x)
        x = self.se(x)
        x = self.arfb3(x)
        x = ori+x
        return x
        

class Config():
    lamRes = torch.nn.Parameter(torch.ones(1))
    lamX = torch.nn.Parameter(torch.ones(1))



if __name__ == "__main__":
    # RU = ResidualUnit(nFeats=4)
    x = torch.tensor([float(i+1) for i in range(2048)]).reshape((1, -1, 4, 4))
    reScale = Config()
    print(x.shape)


    hpb = HPB(x.shape[1], x.shape[1], reScale)
    res = hpb(x)
    # print(res.shape)
    # RU = ResidualUnit(x.shape[1])
    # res = RU(x)
        
