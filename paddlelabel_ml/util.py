import importlib
from pathlib import Path

import yaml
import connexion

HERE = Path(__file__).parent.absolute()


def get_models(model_fdr="model"):
    model_idxs = importlib.resources.contents(f"paddlelabel_ml.{model_fdr}")
    model_idxs = [idx for idx in model_idxs if idx[0] != "_" and idx[0] != "."  ]
    model_idxs.remove("base")

    models = {}
    for idx in model_idxs:
        info = yaml.safe_load(open(HERE / Path(model_fdr) / Path(idx) / Path("info.yaml"), "r").read())
        models[info["name"]] = {
            "path": f"paddlelabel_ml.{model_fdr}.{idx}",
            # "task_categories": info["task_categories"],
            "interactive": info.get("interactive", False),
            **info,
        }
    return models


def abort(detail, status, title=""):
    raise connexion.exceptions.ProblemException(detail=detail, title=title, status=status, headers={"message": detail})
