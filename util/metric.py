import torch
from pytorch_msssim import ssim
import torch.nn.functional as F


def topk_accuracy(output, target, top_k=(1, 5)):
    max_k = max(top_k)
    _, pred = output.topk(max_k, 1, True, True)
    correct = pred.eq(target.view(target.size(0), -1).expand_as(pred))
    return [correct[:, :k].sum().item() for k in top_k]


def get_psnr(output, target, max_pixel=255.0):
    # [-1, 1] -> [0, 255]
    output = output * 127.5 + 127.5
    target = target * 127.5 + 127.5

    mse = F.mse_loss(output, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel / mse.sqrt())
    return psnr.item()


def get_ssim(output, target, data_range=255.0):
    # [-1, 1] -> [0, 255]
    output = output * 127.5 + 127.5
    target = target * 127.5 + 127.5
    return ssim(output, target, data_range=data_range).item()
