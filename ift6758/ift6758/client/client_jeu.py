import pandas as pd
import json

import sys
import os
#current_dir = os.path.abspath(os.getcwd())
#parent_dir = os.path.dirname(current_dir)
#sys.path.append(os.path.join(parent_dir, 'data'))

from  data.API_features import *

#from ift6758.ift6758.data. API_features import *

class GameClient:
    def __init__(self) -> None:
        self.gameId = 0
        self.last_eventId = -1
        self.game_ended = False

    def process_query(self, gameId):
        if self.game_ended and self.gameId == gameId:
            return None

        # Charger les données JSON
        json_data = API_features.nhl_play_by_play_modified(gameId)

        # Vérifier si le jeu est terminé
        self.game_ended = any(play['typeDescKey'] == "game-end" for play in json_data['plays'])

        if self.game_ended and self.gameId == gameId:
            return None

        # Extraire les nouveaux événements depuis le dernier eventId connu
        new_events = [play for play in json_data['plays'] if play['eventId'] > self.last_eventId]

        if self.gameId != gameId or new_events:
            self.gameId = gameId
            self.last_eventId = new_events[-1]['eventId'] if new_events else self.last_eventId

            # Ajouter les nouveaux événements au fichier JSON
            json_data['newPlays'] = new_events

            df = API_features.features(json_data)
            df = df.drop(columns=['is_goal', 'empty_goal'])
            
            return json_data, df     
        else:
            return None