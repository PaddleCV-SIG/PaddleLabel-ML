from pathlib import Path
import argparse
import importlib

import connexion
from flask_cors import CORS
import paddle

from paddlelabel_ml import util

HERE = Path(__file__).parent.absolute()


def parse_args():
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
    parser.add_argument(
        "--cpu",
        default=False,
        action="store_true",
        help="Force cpu mode, if not set, will use gpu if avaliable",
    )
    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()

    return args


def run():
    args = parse_args()

    connexion_app = connexion.App("paddlelabel_ml")
    connexion_app.add_api(
        HERE / "openapi.yml",
        strict_validation=True,
        pythonic_params=True,
    )
    connexion_app.add_error_handler(Exception, util.backend_error)

    CORS(connexion_app.app)

    host = "0.0.0.0" if args.lan else "127.0.0.1"

    if args.cpu:
        use_gpu = False
    else:
        has_gpu = "gpu" in paddle.device.get_device()
        use_gpu = has_gpu
    util.use_gpu = use_gpu

    connexion_app.run(host=host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    run()


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
