from util.register import REGISTER
from .cls_trainer import CLSTrainer
from .ir_trainer import IRTrainer
from .syn_trainer import SynTrainer

TRAINERS = REGISTER("trainers")
TRAINERS.register_module(CLSTrainer)
TRAINERS.register_module(IRTrainer)
TRAINERS.register_module(SynTrainer)
