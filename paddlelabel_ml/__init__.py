import pathlib

import paddlelabel_ml.model

HERE = pathlib.Path(__file__).parent

version = open((HERE / "version"), "r").read().strip()
