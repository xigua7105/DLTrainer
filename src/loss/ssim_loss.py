import torch
import torch.nn as nn
from pytorch_msssim import ssim


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-1.0, -1.0, -1.0])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([2.0, 2.0, 2.0])[None, :, None, None])

    def forward(self, inp):
        # inp [-1, 1]
        return (inp - self.shift) / self.scale


class SSIMLoss(nn.Module):
    def __init__(self, data_range=(-1, 1), size_average=True):
        super(SSIMLoss, self).__init__()
        assert data_range in [(0, 1), (-1, 1)]
        self.scaler = ScalingLayer() if data_range == (-1, 1) else nn.Identity()
        self.size_average = size_average

    def forward(self, output, target):
        output = self.scaler(output)
        target = self.scaler(target)
        return 1 - ssim(output.float(), target.float(), data_range=1.0, size_average=self.size_average)
