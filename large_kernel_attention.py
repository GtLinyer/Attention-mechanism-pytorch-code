from torch import nn


class LargeKernelAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, (5, 5), padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, (7, 7), stride=(1, 1), padding=9, groups=dim, dilation=(3, 3))
        self.conv1 = nn.Conv2d(dim, dim, (1, 1))

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn
