from model_archs import common
from model_archs import attention
import torch.nn as nn
import torch
from thop import profile

# def make_model(args, parent=False):
#     if args.dilation:
#         from model import dilated
#         return NLSN(args, dilated.dilated_conv)
#     else:
#         return NLSN(args)


class NLSN(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(NLSN, self).__init__()

        n_resblock = 32
        n_feats = 256
        kernel_size = 3 
        scale = 4
        act = nn.ReLU(True)

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [attention.NonLocalSparseAttention(
            channels=n_feats, chunk_size=144, n_hashes=4, reduction=4, res_scale=0.1)]

        for i in range(n_resblock):
            m_body.append( common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ))
            if (i+1)%8==0:
                m_body.append(attention.NonLocalSparseAttention(
                    channels=n_feats, chunk_size=144, n_hashes=4, reduction=4, res_scale=0.1))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, 3, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
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
    model = NLSN().cuda()
    flops, params = profile(model, inputs=(input,))
    print("Param: {} M".format(params/1e6))
    print("FLOPs: {} G".format(flops/1e9))



