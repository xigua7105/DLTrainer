import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights, VGG19_Weights


class ScalingLayer1(nn.Module):
    def __init__(self):
        super(ScalingLayer1, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        # inp [-1, 1]
        return (inp - self.shift) / self.scale


class ScalingLayer2(nn.Module):
    def __init__(self):
        super(ScalingLayer2, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, inp):
        # inp [0, 1]
        return (inp - self.shift) / self.scale


class VGG(nn.Module):
    def __init__(self, vgg_name, feature_layers, requires_grad=False):
        super().__init__()
        assert vgg_name in ['vgg16', 'vgg19']
        assert len(feature_layers) == 5
        if vgg_name == 'vgg16':
            vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        else:
            vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.DEFAULT).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(feature_layers[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(feature_layers[0], feature_layers[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(feature_layers[1], feature_layers[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(feature_layers[2], feature_layers[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(feature_layers[3], feature_layers[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class Vgg16(VGG):
    def __init__(self, requires_grad=False):
        super().__init__(vgg_name='vgg16', feature_layers=(4, 9, 16, 23, 30), requires_grad=requires_grad)


class Vgg19(VGG):
    def __init__(self, requires_grad=False):
        super().__init__(vgg_name='vgg19', feature_layers=(2, 7, 12, 21, 30), requires_grad=requires_grad)


class PerceptualLoss(nn.Module):
    def __init__(self, data_range=(-1, 1), vgg_name='vgg16', layer_weights=None):
        super().__init__()
        assert data_range in [(0, 1), (-1, 1)]
        self.scaler = ScalingLayer1() if data_range == (-1, 1) else ScalingLayer2()

        self.vgg = Vgg16() if vgg_name == 'vgg16' else Vgg19()
        self.weights = layer_weights if layer_weights is not None else (1.0, 1.0, 1.0, 1.0, 1.0)
        self.n_layers = len(self.weights)

    def forward(self, output, target):
        output_vgg = self.vgg(self.scaler(output))
        with torch.no_grad():
            target_vgg = self.vgg(self.scaler(target))

        perc_loss = 0.0
        for i in range(self.n_layers):
            perc_loss += self.weights[i] * F.l1_loss(output_vgg[i], target_vgg[i])

        return perc_loss
