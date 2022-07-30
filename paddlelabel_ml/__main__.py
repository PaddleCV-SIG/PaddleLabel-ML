import os.path as osp

import connexion
from flask_cors import CORS

basedir = osp.abspath(osp.dirname(__file__))
# workspace_dir = "/home/lin/Desktop/data/pplabel-ml/"

connexion_app = connexion.App("pplabel_ml")

connexion_app.add_api(
    "openapi.yml",
    # request with undefined param returns error, dont enforce body
    strict_validation=True,
    pythonic_params=True,
)

CORS(connexion_app.app)

connexion_app.run(host="0.0.0.0", port=1234, debug=True)

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
