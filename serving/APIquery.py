import numpy as np
import requests
import json

# URL for the predict endpoint
urlLogs = 'http://127.0.0.1:5000/logs'
urlDownload = 'http://127.0.0.1:5000/download_registry_model'
urlPredict= 'http://127.0.0.1:5000/predict'

# Sample data to send in the POST request
data = {
    'comet-workspace': 'me-pic',
    'comet-project': 'milestone-2',
    'model-name': 'logistic_reg_distance',
    'model-version': '1.1.0',
    'data': {
        'shot_distance': (np.random.rand(10) * 20).tolist(),
        'shot_angle': (np.random.rand(10) * 20).tolist(),
    }
}
json_data = json.dumps(data)
headers = {'Content-Type': 'application/json'}
# Predict
headers = {'Content-Type': 'application/json'}
response = requests.post(urlPredict, data=json_data, headers=headers)
# Check the response
if response.status_code == 200:
    result = response.json()
    print("Prediction result:", result)
else:
    print(f"Error: {response.status_code}")
    print(response.text)



data2 = {
    'comet-workspace': 'me-pic',
    'comet-project': 'milestone-2',
    'model-name': 'r-gression-logistique-entrain-sur-la-distance-et-l-angle',
    'model-version': '1.1.0',
    'data': {
        'shot_distance': (np.random.rand(10) * 20).tolist(),
        'shot_angle': (np.random.rand(10) * 20).tolist(),
    }
}
json_data = json.dumps(data2)

# CHANGE MODEL
response = requests.post(urlDownload, data=json_data, headers=headers)
if response.status_code == 200:
    result = response.json()
    print("Prediction result:", result)
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Predict
headers = {'Content-Type': 'application/json'}
response = requests.post(urlPredict, data=json_data, headers=headers)
if response.status_code == 200:
    result = response.json()
    print("Prediction result:", result)
else:
    print(f"Error: {response.status_code}")
    print(response.text)