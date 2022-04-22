from werkzeug.serving import run_simple

from pplabel_ml.__main__ import application

if __name__ == "__main__":
    run_simple(application, use_reloader=True)
