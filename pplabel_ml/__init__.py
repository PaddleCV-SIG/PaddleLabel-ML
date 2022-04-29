import pathlib

import pplabel_ml.model

HERE = pathlib.Path(__file__).parent

version = open((HERE / "version"), "r").read().strip()
