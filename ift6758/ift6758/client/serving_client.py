import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance","Angle"]
        self.features = features

        # any other potential initialization
        self.previous_model = ""
        self.current_model = ""

    def predict(self, X: pd.DataFrame, features) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        if features is not None:
            X = X[features]
        else:
            X = X[self.features]
        json = requests.post(self.base_url + "/predict", json=X.to_json())

        if json.status_code == 200:
            liste = json.content.decode("utf-8")
            liste = liste[1:len(liste) - 1].split(sep=",")
            liste = [float(i) for i in liste]
        else:
            raise Exception(f"The application returned the error {json.status_code}")
        return pd.DataFrame(liste)

    def logs(self) -> dict:
        """Get server logs"""
        log = requests.get(self.base_url + "/logs")
        raise log

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
        self.last_model_name = self.current_model_name
        self.current_model_name = model
        client_choice = {
            "workspace": workspace,
            "model": model,
            "version": version
        }

        response = requests.post(self.base_url + "/download_registry_model", json=client_choice)
        response = response.json()
        return response
