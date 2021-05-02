from flask import Flask, jsonify, request
import numpy as np
import joblib

import flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        lr = joblib.load('model.pkl')
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = np.array(list(map(float, to_predict_list))).reshape(1, -1)
        print(to_predict_list)
        prediction = lr.predict(to_predict_list)
        return jsonify({'prediction': list(prediction)})
    except:
        print("Hello")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
