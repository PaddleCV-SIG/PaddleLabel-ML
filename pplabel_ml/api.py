import time

from connexion import request

from pplabel_ml.model import models
from pplabel_ml.util import abort

global loaded_models
loaded_models = {}


def isRunning():
    return True


def getAll():
    return [{"name": n} for n in models.keys()]



    
def train(model_name):
    # print("train", model_name)
    load(model_name, reload=True)
    model = loaded_models[model_name]
    
    if model.training:
        abort(
            f"Model {model_name} is in training. Can't train this model till current train finishes.",
            500,
        )
    model.training = True
    try:
        model.train(data_dir=request.json["data_dir"])
    except Exception as e:
        raise e
    finally:
        model.training = False


def eval(model_name):
    pass


def getProgress(model_name):
    pass


def predict(model_name):
    if model_name not in loaded_models.keys():
        abort(f"Model {model_name} not loaded, call load endpoint first!", 500)
    return {"result": loaded_models[model_name].predict(request.json)}


def load(model_name, reload=False):
    params = request.json
    if model_name not in models.keys():
        abort(f"No model named {model_name}", 404)
    if model_name not in loaded_models.keys():
        loaded_models[model_name] = models[model_name](**params)
    loaded_models[model_name].load_time = time.time()
    return f"Model {model_name} loaded", 200

def unload(model_name):
    if model_name not in models.keys():
        abort(f"No model named {model_name}", 404)
    if model_name in loaded_models.keys():
        del loaded_models[model_name]
