import json
import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features
        
        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        logger.info("Starting prediction request")
        url = f"{self.base_url}/predict"
        headers = {'Content-Type': 'application/json'}
        data = X.to_json(orient='records')

        try:
            response = requests.post(url, headers=headers, json=json.loads(data))
            if response.status_code == 200:
                logger.info("Prediction successful")
                return pd.DataFrame(response.json())
            else:
                logger.error(f"Failed to get prediction: {response.status_code} {response.text}")
                return pd.DataFrame()
        except Exception as e:
            logger.exception("Exception occurred during prediction request")
            return pd.DataFrame()

    def logs(self) -> dict:
        """Get server logs"""

        logger.info("Fetching server logs")
        url = f"{self.base_url}/logs"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info("Successfully fetched logs")
                return response.json()
            else:
                logger.error(f"Failed to fetch logs: {response.status_code} {response.text}")
                return {}
        except Exception as e:
            logger.exception("Exception occurred while fetching logs")
            return {}


    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        logger.info(f"Requesting download of model {model} from workspace {workspace} with version {version}")
        url = f"{self.base_url}/download_registry_model"
        headers = {'Content-Type': 'application/json'}
        
        data = {
            'comet-workspace': workspace,
            'model-name': model,
            'model-version': version
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                logger.info("Model download successful")
                return response.json()
            else:
                logger.error(f"Failed to download model: {response.status_code} {response.text}")
                return {}
        except Exception as e:
            logger.exception("Exception occurred during model download request")
            return {}
