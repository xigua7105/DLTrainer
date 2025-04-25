from util.register import REGISTER
from .resnet.resnet import ResNet


MODELS = REGISTER("models")
MODELS.register_module(ResNet)
