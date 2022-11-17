import math

from torch import nn


class EfficientChannelAttention(nn.Module):

    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=(k,), padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        wt = self.avg_pool(x)
        wt = self.conv1(wt.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wt = self.sigmoid(wt)
        return x * wt
