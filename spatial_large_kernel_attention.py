from torch import nn

from large_kernel_attention import LargeKernelAttention


class SpatialLargeKernelAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, (1, 1))
        self.activation = nn.GELU()
        self.spatial_gating_unit = LargeKernelAttention(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, (1, 1))

    def forward(self, x):
        short_cut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + short_cut
        return x
