from os import system
from flask import Flask, request, jsonify, send_from_directory
import traceback
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/predictByClientId', methods=['POST'])
def predictByClientId():
    try:
        json_ = request.json
        file1 = open('final_model1.pkl', 'rb')
        model = pickle.load(file1)
        file1.close()
        data_set = pd.read_csv('X_test_2')
        data_set = data_set.drop(['Unnamed: 0', 'TARGET'], axis =1)
        client_id = json_['SK_ID_CURR']
        #client_id = 1234
        client = data_set.query("index == @client_id")
        # Load the test data
        #data_set['SK_ID_CURR'] = data_set['Unnamed: 0']
        #client=data_set[data_set['SK_ID_CURR']==json_['SK_ID_CURR']]
        
        #client=data_set[data_set['index']==json_['index']]
        # Get prediction and probability
        y_pred_test = model.predict(client)
        y_proba_test = model.predict_proba(client)
        
        # Get feature importances
        feature_importances = pd.DataFrame({
            'feature': data_set.columns,
            'importance': model.feature_importances_ / model.feature_importances_.sum()
        }).to_json(orient='records')
        client_dict = client.to_dict('records')[0]
        return jsonify({
            'client': client_dict,
            'prediction': y_pred_test[0],
                'prediction_proba': y_proba_test[0][1],
                'feature_importances': feature_importances
        })

        #return jsonify({
            #'prediction': y_pred_test[0],
            #'prediction_proba': y_proba_test[0][1],
            #'feature_importances': feature_importances
        #})

    except:
        return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')



@app.route('/predict_new', methods=['POST'])
def predict_new():
    try:
        json_ = request.json
        file1 = open('final_model1.pkl', 'rb')
        model = pickle.load(file1)
        file1.close()

        X = pd.DataFrame({
            'AMT_CREDIT': [round(json_['AMT_CREDIT'], 15)],
            'AMT_ANNUITY': [round(json_['AMT_ANNUITY'], 15)],
            'DAYS_BIRTH': [round(json_['DAYS_BIRTH'], 15)],
            'EXT_SOURCE_1': [round(round(json_['EXT_SOURCE_1'], 15), 10)],
            'EXT_SOURCE_2': [round(json_['EXT_SOURCE_2'], 15)],
            'EXT_SOURCE_3': [round(json_['EXT_SOURCE_3'], 15)]
        })

        # Get prediction and probability
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        # Get feature importances
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_ / model.feature_importances_.sum()
        }).to_json(orient='records')

        # Return prediction, probability, and feature importances
        return jsonify({
            'prediction': y_pred[0],
            'prediction_proba': y_proba[0][1],
            'feature_importances': feature_importances
        })


    except:
        return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')


@app.route('/')
def index():
    return "Welcome to the API"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path,
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
