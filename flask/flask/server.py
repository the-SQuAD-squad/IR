import os
import subprocess

from preprocess import Preprocessor
from flask import Flask, request


def create_app():
    app = Flask(__name__)

    with app.app_context():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/daniele.veri.96/ml_api_key.json"
        p = Preprocessor()

        #healthcheck
        os.system('while true; do nc -lp 8080; done &')

        @app.route("/", methods=['POST'])
        def query():
            data=request.json
            answer = p.process(data)
            return answer

        return app
