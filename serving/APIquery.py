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
        'distance': np.random.rand(10).tolist(),
    }
}

#data = {
#    'comet-workspace': 'me-pic',
#    'comet-project': 'milestone-2',
#    'model-name': 'logistic_reg_distance',
#    'model-version': '1.1.0',
#    'data': {
#        'distance': np.random.rand(10).tolist(),
#        'angle': np.random.rand(10).tolist()
#    }
#}

#data = {
#    'comet-workspace': 'me-pic',
#    'comet-project': 'milestone-2',
#    'model-name': 'r-gression-logistique-entrain-sur-la-distance-et-l-angle',
#    'model-version': '1.1.0',
#    'data': {
#            'distance': 5.0,
#            'angle': 10.0
#    }
#}


json_data = json.dumps(data)
headers = {'Content-Type': 'application/json'}
response = requests.post('http://127.0.0.1:5000/predict', data=json_data, headers=headers)


# Check the response
if response.status_code == 200:
    result = response.json()
    print("Prediction result:", result)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
