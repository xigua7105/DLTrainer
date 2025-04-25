from ._register import MODELS


def get_model(cfg):
    model_struct = cfg.model.struct.copy()
    model_name = model_struct.pop('name')
    return MODELS.get_module(model_name)(**model_struct)
