import torch
from torch import nn


def inf(b, h, w):
    return - torch.diag(torch.tensor(float("inf")).cuda().repeat(h), 0).unsqueeze(0).repeat(b * w, 1, 1)


class CCAttention(nn.Module):

    def __init__(self, in_dim):
        super(CCAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batch_size, _, height, width = x.size()
        proj_query = self.query_conv(x)

        proj_query_h = proj_query.permute(0, 3, 1, 2).contiguous().\
            view(m_batch_size * width, -1, height).permute(0, 2, 1)
        proj_query_w = proj_query.permute(0, 2, 1, 3).contiguous().\
            view(m_batch_size * height, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key_h = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batch_size * width, -1, height)
        proj_key_w = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batch_size * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_h = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batch_size * width, -1, height)
        proj_value_w = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batch_size * height, -1, width)

        energy_h = (torch.bmm(proj_query_h, proj_key_h) +
                    inf(m_batch_size, height, width)).view(m_batch_size, width, height, height).permute(0, 2, 1, 3)
        energy_w = torch.bmm(proj_query_w, proj_key_w).view(m_batch_size, height, width, width)

        concat = self.softmax(torch.cat([energy_h, energy_w], 3))

        att_h = concat[:, :, :, 0: height].permute(0, 2, 1, 3).contiguous().view(m_batch_size * width, height, height)

        att_w = concat[:, :, :, height: height + width].contiguous().view(m_batch_size * height, width, width)
        out_h = torch.bmm(proj_value_h,
                          att_h.permute(0, 2, 1)).view(m_batch_size, width, -1, height).permute(0, 2, 3, 1)
        out_w = torch.bmm(proj_value_w,
                          att_w.permute(0, 2, 1)).view(m_batch_size, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_h + out_w) + x
