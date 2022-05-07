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

# def set_cors_headers_on_response(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'X-Requested-With'
#     response.headers['Access-Control-Allow-Methods'] = 'OPTIONS'
#     print("here")
#     return response
# connexion_app.app.after_request(set_cors_headers_on_response)

# class CorsHeaderMiddleware(object):
#     def __init__(self, app):
#         self.app = app

#     def __call__(self, environ, start_response):
#         def custom_start_response(status, headers, exc_info=None):
#             # append whatever headers you need here
#             headers.append(('Access-Control-Allow-Origin', '*'))
#             headers.append(
#                 ('Access-Control-Allow-Headers', 'X-Requested-With')
#             )
#             headers.append(('Access-Control-Allow-Methods', 'OPTIONS'))
#             print("here")
#             return start_response(status, headers, exc_info)

#         return self.app(environ, custom_start_response)
# connexion_app.app.wsgi_app = CorsHeaderMiddleware(connexion_app.app.wsgi_app)

CORS(connexion_app.app)

# connexion_app.run(host="0.0.0.0", port=1234, debug=True)

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from visualdl.server.args import ParseArgs
import visualdl.server.app as vdlApp

args = {"logdir": ".", "public_path": "/visualdl"}
vdlApp = vdlApp.create_app(ParseArgs(**args))

# application = DispatcherMiddleware(vdlApp, {"/model": connexion_app})
application = DispatcherMiddleware(vdlApp, {"/model": connexion_app})


def run():
    run_simple("0.0.0.0", 1234, application, use_reloader=True, threaded=True)


if __name__ == "__main__":
    run()
