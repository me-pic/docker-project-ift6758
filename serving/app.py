"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import pickle
import logging

import numpy as np
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
            app.logger.info(f"{init_model['model-name']} was successfully loaded...")
            cache.set('loaded_model', loaded_model)
            cache.set('loaded_model_name', init_model["model-name"])
            app.logger.info(f'{init_model["model-name"]} was cached ...')

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
    def GetModelFilePath(model_name):
        if model_name == "logistic_reg_distance":
            return f"models/{json['model-name']}/{json['model-name']}.pickle"
        elif model_name == "r-gression-logistique-entrain-sur-la-distance-et-l-angle":
            return f"models/{json['model-name']}/bmodel_3.pickle"


    # Get POST json data
    json = request.get_json()
    app.logger.info("---------------------- Download Registry Model START ----------------------")
    app.logger.info(json)

    response = json['model-name']

    # The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.
    comet_api_key = os.environ.get("COMET_API_KEY", None)

    # check to see if the model you are querying for is already downloaded
    model_file_path = GetModelFilePath(json['model-name'])

    if os.path.exists(model_file_path):
        app.logger.info(f"{json['model-name']} exists ...")
        # if yes, load that model and write to the log about the model change.
        with open(model_file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            app.logger.info(f"{json['model-name']} was successfully loaded...")
            cache.set('loaded_model', loaded_model)
            cache.set('loaded_model_name', json["model-name"])
            app.logger.info(f'{json["model-name"]} was cached ...')
            response += "was successfully loaded..."

    else :
        try:
            # Si non, essayez de télécharger le modèle:
            output_path = f"./models/{json['model-name']}"
            os.makedirs("./models", exist_ok=True)
            comet_api.download_registry_model(workspace=json['comet-workspace'],
                                              registry_name=json['model-name'],
                                              version=json['model-version'],
                                              output_path=output_path,
                                              expand=True)
            # Si ca réussit, chargez ce modèle et écrivez dans les logs que le modèle change
            app.logger.info(f"{json['model-name']} was successfully downloaded ...")
            with open(model_file_path, 'rb') as file:
                loaded_model = pickle.load(file)
                app.logger.info(f"{json['model-name']} was successfully loaded...")
                cache.set('loaded_model', loaded_model)
                cache.set('loaded_model_name', json["model-name"])
                app.logger.info(f'{json["model-name"]} was cached ...')
                response += "was successfully loaded..."

        except Exception as e:
            # En cas d'échec, écrivez dans les logs qu'il y a eu un échec et conservez le modèle actuellement chargé
            app.logger.info("Error encountered ...", e)
            app.logger.info("KEEPING CURRENT LOADED MODEL...", )

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    app.logger.info("---------------------- Download Registry Model END ----------------------")
    app.logger.info(" RESPONSE : " + response)
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
    app.logger.info("---------------------- Predict Model START ----------------------")
    # Get POST json data
    json_data = request.get_json()
    data = json_data['data']
    app.logger.info(F"JSON DATA : {data}")
    df = pd.DataFrame(data)
    app.logger.info(F"DF : {df}")
    response = None

    try:
        model = cache.get('loaded_model')
        model_name = cache.get('loaded_model_name')
        app.logger.info("model loaded ...")

        if model_name == "logistic_reg_distance":
            app.logger.info("logistic_reg_distance loaded ...")
            # on ramasse seulement le data['distance']
            probs = model.predict_proba(df)
            app.logger.info(f"probs : {probs}")
            response = probs.tolist()

        elif model_name == "r-gression-logistique-entrain-sur-la-distance-et-l-angle":
            app.logger.info("model name : ", model_name)
            # SI cest DISNTANCE-ANGLE, on ramasse le data['distance'] et data['angle']
            app.logger.info("New DF : ", df)



        # TODO: on veut verifier le model quon a de loader
        #   (on pourrait cache le nom du modele a chaque fois quon le load afin de pouvoir faire le check ici)
        # TODO: Si cest DISTANCE,
        # TODO: SINON,

        # TODO: ON VEUT PREDIRE POUR UN DF COMPLET ET NON JUSTE UN

    except Exception as e:
        app.logger.info("Error encountered ...", e)




    app.logger.info("---------------------- Predict Model END ----------------------")
    app.logger.info("response : ")
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


if __name__ == '__main__':
    app.run(debug=True)