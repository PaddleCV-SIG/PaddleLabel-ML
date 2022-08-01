import os.path as osp
import argparse

import connexion
from flask_cors import CORS


parser = argparse.ArgumentParser(description="PP Label")
parser.add_argument(
    "--lan",
    default=False,
    action="store_true",
    help="Whether to expose the service to lan",
)
parser.add_argument(
    "--port",
    default=1234,
    type=int,
    help="The port to use",
)
args = parser.parse_args()


basedir = osp.abspath(osp.dirname(__file__))
# workspace_dir = "/home/lin/Desktop/data/pplabel-ml/"

connexion_app = connexion.App("paddlelabel_ml")

connexion_app.add_api(
    "openapi.yml",
    # request with undefined param returns error, dont enforce body
    strict_validation=True,
    pythonic_params=True,
)

CORS(connexion_app.app)

host = "0.0.0.0" if args.lan else "127.0.0.1"

connexion_app.run(host=host, port=args.port, debug=True, threaded=True)

# from werkzeug.middleware.dispatcher import DispatcherMiddleware
# from werkzeug.serving import run_simple

# from visualdl.server.args import ParseArgs
# import visualdl.server.app as vdlApp

# args = {"logdir": ".", "public_path": "/visualdl"}
# vdlApp = vdlApp.create_app(ParseArgs(**args))

# # application = DispatcherMiddleware(vdlApp, {"/model": connexion_app})
# application = DispatcherMiddleware(vdlApp, {"/model": connexion_app})


# def run():
#     run_simple("0.0.0.0", 1234, application, use_reloader=True, threaded=True)


# if __name__ == "__main__":
#     run()
