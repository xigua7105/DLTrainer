import torch.nn as nn
from util.register import REGISTER
from .perceptual_loss import PerceptualLoss
from .ssim_loss import SSIMLoss
from .ir_loss import IRLoss

default_loss = ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'BCELoss', 'L1Loss', 'MSELoss', 'KLDivLoss', 'SmoothL1Loss']
LOSS_FN = REGISTER("loss")
for loss_name in default_loss:
    loss_fn = getattr(nn, loss_name, None)
    LOSS_FN.register_module(loss_fn, loss_name)

LOSS_FN.register_module(PerceptualLoss)
LOSS_FN.register_module(SSIMLoss)
LOSS_FN.register_module(IRLoss)
