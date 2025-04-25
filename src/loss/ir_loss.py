import torch.nn as nn
from .perceptual_loss import PerceptualLoss
from .ssim_loss import SSIMLoss


class IRLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.rec_loss = nn.L1Loss()
        self.perc_loss = PerceptualLoss(data_range=(-1, 1), vgg_name='vgg16', layer_weights=None)
        self.ssim_loss = SSIMLoss()
        self.weights = weights if weights is not None else (1.0, 1.0, 1.0)

    def forward(self, output, target):
        rec_loss = self.rec_loss(output, target)
        perc_loss = self.perc_loss(output, target)
        ssim_loss = self.ssim_loss(output, target)
        loss = self.weights[0]*rec_loss + self.weights[1]*perc_loss + self.weights[2]*ssim_loss
        return dict(loss=loss, rec_loss=rec_loss, perc_loss=perc_loss, ssim_loss=ssim_loss)
