import time
import importlib
import copy

from connexion import request

from paddlelabel_ml.util import abort, get_models
import paddlelabel_ml

# TODO: switch to a thread safe approach
global loaded_models
global loading_models
loaded_models = {}
loading_models = set()

models = get_models()


def isBackendUp():
    return True


def getVersion():
    return paddlelabel_ml.version


def getAll():
    res = copy.deepcopy(models)
    for name in res.keys():
        del res[name]["path"]
    return [{"name": k, **v} for k, v in res.items()]


def getProgress():
    pass


def train(model_name):
    load(model_name, reload=True)  # TODO: this is causing bug?
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


def evaluate(model_name):
    pass


def getProgresss(model_name):
    pass


async def load(model_name, reload=False):
    tic = time.time()

    params = request.json.get("init_params", {})

    # 1. ensure backend has this model
    if model_name not in models.keys():
        abort(f"No model named {model_name}", 404)

    # 2. model is loading or loaded
    if model_name in loaded_models.keys():
        # 2.1 loading
        if model_name in loading_models:
            abort(f"Model {model_name} is still loading, check back after 1 or 2 minutes!", 500)
        # 2.2 loaded with same params
        if loaded_models[model_name].params == params:
            return f"Model {model_name} is already loaded", 200

    # 3. load model
    try:
        model = importlib.import_module(models[model_name]["path"])
        loading_models.add(model_name)
        loaded_models[model_name] = model.Model(**params)
        loaded_models[model_name].params = params
    except Exception as e:
        abort(f"Model load failed. {str(e)}", 500)
    loading_models.remove(model_name)

    loaded_models[model_name].load_time = time.time()
    print(f"Load model {model_name} took {time.time() - tic} s")
    return f"Model {model_name} loaded", 200


async def predict(model_name):
    tic = time.time()
    if model_name not in loaded_models.keys():
        abort(f"Model {model_name} not loaded, call load endpoint first!", 500)
    if model_name in loading_models:
        abort(f"Model {model_name} is still loading, check back after 1 or 2 minutes!", 500)

    params = request.json
    if "piggyback" in params.keys():
        piggy_value = params["piggyback"]
        del params["piggyback"]
    else:
        piggy_value = None

    res = {"predictions": loaded_models[model_name].predict(request.json)}
    if piggy_value is not None:
        res["piggyback"] = piggy_value
    print(f"Inference took {time.time() - tic} s")
    return res


def unload(model_name):
    if model_name not in models.keys():
        abort(f"No model named {model_name}", 404)
    if model_name in loaded_models.keys():
        del loaded_models[model_name]
        return f"Model {model_name} unloaded!"
    return f"Model {model_name} is not loaded!"
