from preprocess import Preprocessor
from flask import Flask, request,jsonify
import os
app = Flask(__name__)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/daniele.veri.96/ml_api_key.json"
p = Preprocessor()

@app.route("/", methods=['POST'])
def hello():
    data=request.json
    answer = p.process(data)
    return answer

if __name__ == "__main__":
    app.run()
