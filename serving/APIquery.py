import requests
import json

# URL for the predict endpoint
url = 'http://127.0.0.1:5000/logs'
urlDownload = 'http://127.0.0.1:5000/download_registry_model'

# Sample data to send in the POST request
data = {
    'comet-workspace': 'me-pic',
    'comet-project': 'milestone-2',
    'model-name': 'logistic_reg_distance',
    'model-version': '1.1.0'
}

# Convert the data to JSON format
json_data = json.dumps(data)

# Set the headers for the POST request
headers = {'Content-Type': 'application/json'}

# Make the POST request
#response = requests.get(url, data=json_data, headers=headers)
response = requests.post(urlDownload, data=json_data, headers=headers)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("Prediction result:", result)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
