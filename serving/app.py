"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import pickle
import logging
import xgboost
from flask import Flask, request, jsonify
from flask_caching import Cache
import pandas as pd
from comet_ml import *

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 0
}
api_key = "K02RXr1jocA72JaixT6WZnR5s"
init_model = {
    'comet-workspace': 'me-pic',
    'comet-project': 'milestone-2',
    'model-name': 'logistic_reg_distance',
    'model-version': '1.1.0'
}


app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app=app)

# On start
with app.app_context():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    app.logger.info("Init ...")

    # TODO: any other initialization before the first request (e.g. load default model)
    try:
        comet_api = API(api_key)
        cache.set('comet_api', comet_api)
        output_path = f"./models/{init_model['model-name']}"
        # On charge le modele initial de comet.ml
        os.makedirs("./models", exist_ok=True)
        comet_api.download_registry_model(workspace=init_model['comet-workspace'],
                                          registry_name=init_model['model-name'],
                                          version=init_model['model-version'],
                                          output_path=output_path,
                                          expand=True)

        file_dir = os.listdir(f'./models/{init_model["model-name"]}')
        file_name = os.path.join(output_path, file_dir[0])
        with open(file_name, 'rb') as file:
            loaded_model = pickle.load(file)
            print(file_name)
            cache.set('loaded_model', loaded_model)
            app.logger.info(f"loaded model : {init_model['model-name']}.")
    except Exception as e:
        app.logger.info("Error encountered ...", e)
    app.logger.info("Init End ...")

# http://127.0.0.1:5000/logs
@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    with open(LOG_FILE) as file:
        logs = file.read().splitlines()
    response = logs
    return jsonify(response)  # response must be json serializable!



# Download_registry_model : Encapsule essentiellement la même fonction dans l'API comet.ml pour télécharger un modèle et
# mettre à jour le modèle actuellement chargé
# Vous devez:
    # Vérifier si le modèle que vous cherchez est déjà téléchargé
        # Si oui, chargez ce modèle et écrivez dans les logs que le modèle change
        # Si non, essayez de télécharger le modèle:
            # Si ca réussit, chargez ce modèle et écrivez dans les logs que le modèle change
            # En cas d'échec, écrivez dans les logs qu'il y a eu un échec et conservez le modèle actuellement chargé

# http://127.0.0.1:5000/download_registry_model
@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info("Download Registry Model Start ... ")
    app.logger.info(json)

    json['model_name'] = json
    # The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.
    comet_api_key = os.environ.get("COMET_API_KEY", None)

    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    raise NotImplementedError("TODO: implement this endpoint")

    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!




# Prédit la probabilité que le tir soit un but compte tenu des inputs
# Input: Les caractéristiques utilisées par votre modèle, compatibles avec model.predict() ou model.predict_proba()
    # indice: df.to_json()
# Sortie: Les prédictions pour toutes les caractéristiques; le output
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO:
    raise NotImplementedError("TODO: implement this enpdoint")
    
    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

if __name__ == '__main__':
    app.run(debug=True)