import os
from serrelabmodels.utils import py_utils

def get_model(cfg):
    print('importing ',cfg['import_prepath'], '.',cfg['import_class'])
    model_module = py_utils.import_module(cfg['import_prepath'])
    model = getattr(model_module, cfg['import_class'])(**cfg['args'])
    return model