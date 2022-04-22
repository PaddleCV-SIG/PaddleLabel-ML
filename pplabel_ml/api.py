import json
import time

from connexion import request

from pplabel_ml.model import models
from pplabel_ml.util import abort

global loaded_models
loaded_models = {}

def getAll():
    print()
    return json.dumps({"name": "clas pdx"})

def train(model_name):
    model = load(model_name)
    if model.training:
        abort(f"Model {model_name} is in training", 500)
    model.training = True
    model.train(data_dir=request.json['data_dir'])
    model.training = False

def eval(model_name):
    pass

def getProgress(model_name):
    pass

def predict(model_name):
    return json.dumps({"result":load(model_name).predict(request.json['img'])} )

def load(model_name):
    if model_name not in models.keys():
        abort(f"Model {model_name} not found", 404)
    if model_name not in loaded_models.keys():
        loaded_models[model_name] = models[model_name]()
    loaded_models[model_name].load_time = time.time()
    return loaded_models[model_name]

def unload(model_name):
    if model_name not in models.keys():
        abort(f"Model {model_name} not found", 404)
    if model_name in loaded_models.keys():
        del loaded_models[model_name]