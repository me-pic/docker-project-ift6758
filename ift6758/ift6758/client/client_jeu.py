import pandas as pd
import numpy as np

from ift6758.data.API_features import *

class GameClient:
    def __init__(self) -> None:
        self.gameId = 0
        self.Tracker = None
        
        
        
    def process_query(self, gameId, model_name="r-gression-logistique-entrain-sur-la-distance-et-l-angle"):
        """
        Produces a dataframe with the features required by the XGBoost model.
        Returns None if the game has already been fully processed, or if the updated
            records do not contain data (e.g. breaks, period changes, etc.)
        Returns a dataframe if valid records are found.
        """
        
        if self.model_name != model_name: self.gameId = 0
        self.model_name = model_name
        
        # load game
        df = retrieve_nhl_play_by_play_modified(gameId)
        
        # if same game, slice. last_eventIdx was set in the last call
        if self.gameId == gameId:
            df = df.iloc[self.Tracker+1:] 
            if len(df) == 0: return None
            
        #self.game_ended = "GAME_END" in df["eventType"].values
        self.Tracker = df.index[-1]


        self.gameId = gameId
        if model_name == 'r-gression-logistique-entrain-sur-la-distance-et-l-angle':
            return df 
        else : return df.drop('angle', axis=1)
    