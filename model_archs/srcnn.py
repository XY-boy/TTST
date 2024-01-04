import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from thop import profile

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)

        return out

if __name__ == '__main__':
    input = torch.rand(1, 3, 128, 128).cuda()  # B C H W
    model = SRCNN().cuda()
    flops, params = profile(model, inputs=(input,))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))

    output = model(input)
    print(output.size())