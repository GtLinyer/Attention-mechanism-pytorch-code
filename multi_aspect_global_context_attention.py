import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MultiAspectGCAttention(nn.Module):

    def __init__(self, inplanes, ratio, headers=1, pooling_type='att', fusion_type='channel_add'):
        super(MultiAspectGCAttention, self).__init__()
        assert pooling_type in ['avg', 'att']

        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        assert inplanes % headers == 0 and inplanes >= 8  # inplanes must be divided by headers evenly

        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = False

        self.single_header_inplanes = int(inplanes / headers)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=(1, 1))
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1)),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1)))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1)),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1)))
            # for concat
            self.cat_conv = nn.Conv2d(2 * self.inplanes, self.inplanes, kernel_size=(1, 1))
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1)),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1)))

    def _spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            x = x.view(batch * self.headers, self.single_header_inplanes, height, width)
            input_x = x

            input_x = input_x.view(batch * self.headers, self.single_header_inplanes, height * width)

            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch * self.headers, 1, height * width)

            # scale variance
            if self.att_scale and self.headers > 1:
                single_header_inplanes = torch.from_numpy(np.array(self.single_header_inplanes))
                context_mask = context_mask / torch.sqrt(single_header_inplanes)

            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self._spatial_pool(x)
        out = x

        if self.fusion_type == 'channel_mul':
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, c1, _, _ = channel_concat_term.shape
            n, c2, h, w = out.shape

            out = torch.cat([out, channel_concat_term.expand(-1, -1, h, w)], dim=1)
            out = self.cat_conv(out)
            out = F.layer_norm(out, [self.inplanes, h, w])
            out = F.relu(out)
        return out
