from torch import nn
from torch.nn import functional as F

from channel_attention import ChannelAttention
from spatial_attention import SpatialAttention


class CBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, _stride=(1, 1)):
        super(CBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), stride=_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.channel = ChannelAttention(self.expansion*planes)
        self.spatial = SpatialAttention()

        self.shortcut = nn.Sequential()
        if _stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=(1, 1), stride=_stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        cbam_c_out = self.channel(out)
        out = out * cbam_c_out
        cbam_s_out = self.spatial(out)
        out = out * cbam_s_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out
