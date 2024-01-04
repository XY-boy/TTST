import torch
import torch.nn as nn

class PrintLayer(nn.Module):
    def __init__(self, x=''):
        super().__init__()
        self.msg = x

    def forward(self, x):
        print(self.msg, x.shape)
        return x

# High Filter Module
class HFM(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        
        self.k = k

        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size = self.k, stride = self.k),
            nn.Upsample(scale_factor = self.k, mode = 'nearest'),
        )

    def forward(self, tL):
        assert tL.shape[2] % self.k == 0, 'h, w must divisible by k'
        return tL - self.net(tL)


if __name__ == '__main__':
    m = HFM(2)
    x = torch.tensor([float(i+1) for i in range(16)]).reshape((1, 1, 4, 4))
    y = m(x)
