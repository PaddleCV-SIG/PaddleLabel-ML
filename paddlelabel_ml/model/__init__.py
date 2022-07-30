import pkgutil

from paddlelabel_ml.model.base.model import BaseModel
from . import util

models = {}


def add_model(model):
    models[model.name] = model


# load all models
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    _ = loader.find_module(module_name).load_module(module_name)
