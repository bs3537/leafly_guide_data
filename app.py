from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import pickle 
import joblib
import json
import numpy as np
import requests

#https://www.kdnuggets.com/2019/10/easily-deploy-machine-learning-models-using-flask.html

app = Flask(__name__)

def create_app():
    model = joblib.load(open('potguide_model.pkl', 'rb'))

#################APP ROUTES####################

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predicttest',methods=['POST'])
    def predicttest():

        str_features = [str(x) for x in request.form.values()]
        features = [np.array(str_features)]

        data = [np.array(features)]
        model = joblib.load(open('potguide_model.pkl', 'rb'))
        #prediction = np.array2string(model.predict(data))
        prediction = np.array(model.predict(data))
        #output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='The recommended cannabis strain is {}'.format(prediction))

    @app.route('/predict', methods=['POST', 'GET'])
    def predict():
        model = joblib.load('potguide_model.pkl')
        d = request.form.values()
        #data = d.json()
        prediction = np.array2string(model.predict(d))
        res = jsonify(prediction)
        return render_template('index.html', prediction_text='The recommended cannabis strain is {}'.format(prediction))



    @app.route('/results',methods=['POST'])
    def results():

        data = request.get_json(force=True)
        model = joblib.load(open('potguide_model.pkl', 'rb'))
        prediction = np.array2string(model.predict(data))
        return jsonify(prediction)
        
    return app

if __name__ == "__main__":
    app.run(debug=True)