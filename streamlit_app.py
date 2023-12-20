import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import requests

from ift6758.ift6758.client.client_jeu import *
from ift6758.ift6758.client.serving_client import *


"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""



IP = os.environ.get("SERVING_IP", "0.0.0.0")
PORT = os.environ.get("SERVING_PORT", 5001)
base_url = f"http://{IP}:{PORT}"



st.title("Hockey Visualisation App")


if 'gameClient' not in st.session_state:
    Client_jeu = GameClient()
    st.session_state['gameClient'] = Client_jeu

if 'servingClient' not in st.session_state:
    client = ServingClient(ip=IP, port=PORT)
    st.session_state['servingClient'] = client

if 'model_downloaded' not in st.session_state:
     st.session_state['model_downloaded'] = False

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'model_selection_change' not in st.session_state:
    st.session_state['model_selection_change'] = False

if 'stored_df' not in st.session_state: 
    st.session_state.stored_df = None

if 'pred_goals' not in st.session_state:
    st.session_state.pred_goals = [0,0]
    
if 'real_goals' not in st.session_state:
    st.session_state.real_goals = [0,0]

if 'teams' not in st.session_state:
    st.session_state.teams = None

if 'teams_full' not in st.session_state:
    st.session_state.teams_full = None

### functions
def get_play_by_play(game_id: str):

        """
        If file doesn't exist: Downloads play-by-play data for the give game ID and returns the raw json content
        If file already exists: Skips download and returns the raw json content
        The file is saved in json format at path/<game_id>
        Args:
            game_id (str): Game ID for play-by-play data according to the format documented
                in https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
            path: Path to which the file is saved
            use_cache: whether we check for existence of file.
        Returns: raw data for the specific game in the form of a json object
        """
        
        response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
        json_response = response.json()

        return json_response

def get_scores(data_never_seen_json: str):
    """
    Get number of real goals from live feed dict (raw data).
    """
    data = get_play_by_play(game_id)
    
    away_team_name = data_never_seen_json['awayTeam']['name']['default']
    away_team_id = data_never_seen_json['awayTeam']['id']
    home_team_name = data_never_seen_json['homeTeam']['name']['default']
    home_team_id = data_never_seen_json['homeTeam']['id']
    
    Team_info = {
        'away_team_name' : away_team_name, 
        'away_team_id' : away_team_id, 
        'home_team_name' : home_team_name, 
        'home_team_id' : home_team_id
    } 


    shots_on_goal = []

    for event in data['plays']:
        if event.get("typeDescKey") == "shot-on-goal":
            details = event.get('details', {})
            shot_info = {
                'away_score': details.get('awaySOG', 0),
                'home_score': details.get('homeSOG', 0),
                'event_owner_team_id': details.get('eventOwnerTeamId')
            }
            shots_on_goal.append(shot_info)

    return shots_on_goal, Team_info



def calculate_game_goals(df: pd.DataFrame, shots_on_goal_info,Team_info, threshold=0.5):
    """
    Sum over model predictions for each team and determine goals based on a threshold.
    
    Input: 
        df (DataFrame): DataFrame with feature values and model predictions.
        shots_on_goal_info (list of dicts): Information about each "shot-on-goal" event.
        threshold (float): Threshold for determining if a predicted shot is a goal.
    
    Output: 
        pred_goals (list): Predicted number of goals for each team.
    """
    # Initialiser les compteurs de buts prédits
    sum_pred_away_team = 0
    sum_pred_home_team = 0

    # Parcourir chaque prédiction
    for idx in range(len(df)):
        model_output = df.at[idx, 'Model Output']
        if model_output > threshold:
            shot_info = shots_on_goal_info[idx]
            if shot_info['event_owner_team_id'] == Team_info['away_team_id']:
                sum_pred_away_team += 1
            elif shot_info['event_owner_team_id'] == Team_info['home_team_id']:
                sum_pred_home_team += 1

    pred_goals = [sum_pred_away_team, sum_pred_home_team]

    return pred_goals

def get_period_info(game_id: str):
    data = get_play_by_play(game_id)
    
    period = None
    periodTimeRemaining = None
    
    if 'plays' in data and data['plays']:

        last_play = data['plays'][-1]
        period = last_play.get("period")
        periodTimeRemaining = last_play.get("timeRemaining")

    return period, periodTimeRemaining


def check_game_end(game_id: str):

    df = get_play_by_play(game_id)


    game_end = "game-end" == df['plays'][-1]['typeDescKey']
    return game_end


#### Streamlit app

with st.sidebar:
    # TODO: Add input for the sidebar

    workspace = st.selectbox(label='Workspace', options=['me-pic'] )
    model = st.selectbox(label='Model', options=['Logistic_reg_distance', 'r-gression-logistique-entrain-sur-la-distance-et-l-angle'])
    version = st.selectbox(label='Model version', options=['1.0.0','1.1.0']) 

    model_button = st.button('Get Model')
    
    # If button click and no model change
    if model_button and model == st.session_state.model: 
        st.write(f':red[Model {model} already chosen!]')
    
    # If button click and model change
    elif model_button: 
        st.session_state['model_downloaded'] = True 
        st.session_state['model'] = model

        try:
            st.session_state.servingClient.download_registry_model(workspace, st.session_state.model, version)
            st.write(f'Got model:\n **{st.session_state.model}**!')
        except Exception as e:
            st.error(f"Erreur lors du téléchargement du modèle : {e}")

        # Reinitialize session state objects if model changes
        st.session_state.stored_df = None 
        st.session_state.real_goals = [0,0]
        st.session_state.pred_goals = [0,0]
        st.session_state.gameClient.gameId = 0

    # If no button click, but page rerun: show previous model
    elif not model_button and st.session_state.model_downloaded:  
        st.write(f'Got model:\n **{st.session_state.model}**!')
    
    # If no button click, and no previous downloaded model
    else: 
        st.write('Waiting on **Get Model** button press...')

with st.container():
    # TODO: Add Game ID input
    st.write("Game ID:")
    game_id = st.text_input(label='Input Game ID:', value='2021020329', max_chars=10, label_visibility='collapsed')
    
    # Reinitialize session state objects if game_id changes
    if game_id != st.session_state.gameClient.gameId: 
        st.session_state.stored_df = None
        st.session_state.real_goals = [0,0]
        st.session_state.pred_goals = [0,0]

    pred_button = st.button('Ping game')
    if pred_button:
        # If no model was downloaded first, ask to download model
        if st.session_state.model_downloaded == False: 
            st.write(':red[Please download model first!]')
        else: 
            st.write(f'**The current game ID is {game_id}!**') 

with st.container():
    # TODO: Add Game info and predictions
    if pred_button and st.session_state.model_downloaded:
        
        # Get dataframe of new events
        data_returned = st.session_state.gameClient.process_query(game_id)
        if data_returned is not None:
            data_never_seen_json, data_never_seen_df = data_returned
        else:
        # Handle the None case, maybe log an error or set default values
            data_never_seen_json, data_never_seen_df = None, None
            print("No more data")
        # If there are new events: 
        if data_never_seen_df is not None: 
            # Make predictions on events 
            pred_MODEL = st.session_state.servingClient.predict(data_never_seen_df)
            
            #df = pd.DataFrame(data_never_seen_df, columns=st.session_state.servingClient.features) 
            #df = df.reset_index(drop=True)
            data_never_seen_df['Model Output'] = pred_MODEL[1]
            # Calculate game actual goals and goal predictions 
            shots_on_goal_info, Team_info = get_scores(data_never_seen_json)
            pred_goals = calculate_game_goals(data_never_seen_df, shots_on_goal_info, Team_info)
            
            real_away_score = 0
            real_home_score = 0

            if shots_on_goal_info:
                latest_shot_info = shots_on_goal_info[-1]
                real_away_score = latest_shot_info['away_score']
                real_home_score = latest_shot_info['home_score']

                # ID des équipes
                away_team_id = Team_info['away_team_id']
                home_team_id = Team_info['home_team_id']

                teams_A_H = [away_team_id, home_team_id]
                for i, team_id in enumerate(teams_A_H):
                    st.session_state.pred_goals[i] += pred_goals[i]
                    if team_id == away_team_id:
                        st.session_state.real_goals[i] = real_away_score
                    else:
                        st.session_state.real_goals[i] = real_home_score

                st.session_state.teams = teams_A_H
                     
            # Gestion de l'identifiant du jeu
            if game_id == st.session_state.gameClient.gameId: 
                df = pd.concat([st.session_state.stored_df, data_never_seen_df], ignore_index=True)
                st.session_state.stored_df = data_never_seen_df 
            else: 
                st.session_state.stored_df = data_never_seen_df 
                
            away_team_name = Team_info['away_team_name']
            home_team_name = Team_info['home_team_name']
            st.subheader( f"Game {game_id} : {away_team_name} VS {home_team_name}")

        period, periodTimeRemaining = get_period_info(game_id)
        if check_game_end(game_id): 
            st.write('**Game ended!**')
            st.write(f'Game end at: **Period:** {period}  --  **Period time remaining:** {periodTimeRemaining}')  
        else: 
            st.write('**Game live!**')        
            st.write(f'**Period:** {period}  --  **Period time remaining:** {periodTimeRemaining}')      

        # Display game goal predictions and info:
        col1, col2 = st.columns(2)
        pred_goals_round = np.round(st.session_state.pred_goals, decimals=1)

        delta1 = float(np.round(pred_goals_round[0] - st.session_state.real_goals[0], decimals=1))
        delta2 = float(np.round(pred_goals_round[1] - st.session_state.real_goals[1], decimals=1))
        col1.metric(label=f"**{st.session_state.teams[0]}** xG (actual)", value=f"{pred_goals_round[0]} ({st.session_state.real_goals[0]})", delta=delta1)
        col2.metric(label=f"**{st.session_state.teams[1]}** xG (actual)", value=f"{pred_goals_round[1]} ({st.session_state.real_goals[1]})", delta=delta2)

    else:
        st.write('Waiting on **Ping Game** button press...')
    
    st.write('')
    st.write('')

    

with st.container():
    # TODO: Add data used for predictions

    
    st.header("Data used for predictions (and predictions)")
    if pred_button and st.session_state.model_downloaded:
        st.write(st.session_state.stored_df)
    else:
        st.write('Waiting on **Ping Game** button press...')



