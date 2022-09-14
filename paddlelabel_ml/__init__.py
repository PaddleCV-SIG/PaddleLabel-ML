import pathlib

import paddlelabel_ml.model

project_base = pathlib.Path(__file__).parent
version = open((project_base / "version"), "r").read().strip()
