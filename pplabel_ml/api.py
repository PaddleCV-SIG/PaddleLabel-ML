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


def getProgresss(model_name):
    pass


def predict(model_name):
    tic = time.time()
    if model_name not in loaded_models.keys():
        abort(f"Model {model_name} not loaded, call load endpoint first!", 500)
    res = {"result": loaded_models[model_name].predict(request.json)}
    print(f"Inference took {time.time() - tic} s")
    return res


def load(model_name, reload=False):
    tic = time.time()
    params = request.json.get("init_params", {})
    # print(models)
    if model_name not in models.keys():
        abort(f"No model named {model_name}", 404)

    if model_name not in loaded_models.keys() or (
        model_name in loaded_models.keys() and loaded_models[model_name].params != params
    ):
        loaded_models[model_name] = models[model_name](**params)
    loaded_models[model_name].params = params

    loaded_models[model_name].load_time = time.time()
    print(f"Load model {model_name} took {time.time() - tic} s")
    return f"Model {model_name} loaded", 200


def unload(model_name):
    if model_name not in models.keys():
        abort(f"No model named {model_name}", 404)
    if model_name in loaded_models.keys():
        del loaded_models[model_name]
        return f"Model {model_name} unloaded!"
    return f"Model {model_name} is not loaded!"
