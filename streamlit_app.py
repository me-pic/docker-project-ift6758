import streamlit as st
import pandas as pd
import numpy as np
import os
import requests


from ift6758.ift6758.client.client_jeu import GameClient
from ift6758.ift6758.client.serving_client import ServingClient


"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""


IP = os.environ.get("SERVING_IP", "127.0.0.1")
PORT = os.environ.get("SERVING_PORT", 5000)
base_url = f"http://{IP}:{PORT}"



st.title("Hockey Visualisation App")


if 'gameClient' not in st.session_state:
    Client_jeu = GameClient()
    st.session_state['gameClient'] = Client_jeu

if 'servingClient' not in st.session_state:
    ServingClient = ServingClient(ip=IP, port=PORT)
    st.session_state['servingClient'] = ServingClient

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

def get_scores(game_id: str):
        """
        Get number of real goals from live feed dict (raw data)
        """

        data = get_play_by_play(game_id)
        goal_a = None # away team
        goal_h = None # home team
        
        if data["liveData"].get("linescore") is not None:
            goal_a = data["liveData"]["linescore"]["teams"]["away"]["goals"]
            goal_h = data["liveData"]["linescore"]["teams"]["home"]["goals"]
        
        # Bug while trying to get number of goals from linescore
        if goal_a is None and goal_h is None: 
            goal_a = data["liveData"]["plays"]["currentPlay"]["about"]["goals"]["away"]
            goal_h = data["liveData"]["plays"]["currentPlay"]["about"]["goals"]["home"]

        # Get away and home teams tricode
        away = data["gameData"]["teams"]["away"]["triCode"]
        home = data["gameData"]["teams"]["home"]["triCode"]

        # Get away and home teams full name
        away_full = data["gameData"]["teams"]["away"]["name"]
        home_full = data["gameData"]["teams"]["home"]["name"]

        real_goals = [goal_a, goal_h]
        teams = [away, home]
        teams_full = [away_full, home_full]  

        return real_goals, teams, teams_full


def calculate_game_goals(df: pd.DataFrame, pred: pd.DataFrame, teams_A_H: list):
    """
    Sum over model_pred for each team 
        Input: 
            df (DataFrame), with feature values 
            pred (DataFrame), model prediction for every event
            teams_A_H (list), teams tricode: [away, home]
        Output: 
            pred_goals (list), predicted number of goals for each team
            teams (list), abbreviation for each team

    """
    df = df.reset_index(drop=True)

    df['Model Output'] = pred

    pred_team1 = df.loc[df['team']==teams_A_H[0] , 'Model Output']
    sum_pred_team1 = pred_team1.sum()
    pred_team2 = df.loc[df['team']==teams_A_H[1] , 'Model Output']
    sum_pred_team2 = pred_team2.sum()

    pred_goals = [sum_pred_team1, sum_pred_team2]

    return pred_goals

def get_period_info( game_id: str):

        data = get_play_by_play(game_id)

        # Get Period and PeriodTimeRemaining
        period = data["plays"][-1]["period"]
        periodTimeRemaining = data["plays"][-1]["timeRemaining"] 

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
    version = st.selectbox(label='Model version', options=['1.0.0']) 

    model_button = st.button('Get Model')
    
    # If button click and no model change
    if model_button and model == st.session_state.model: 
        st.write(f':red[Model {model} already chosen!]')
    
    # If button click and model change
    elif model_button: 
        st.session_state['model_downloaded'] = True 
        st.session_state['model'] = model

        st.session_state.servingClient.download_registry_model(workspace, st.session_state.model, version)
        st.write(f'Got model:\n **{st.session_state.model}**!')

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
    st.header(f"Game goal predictions")
    if pred_button and st.session_state.model_downloaded:
        
        # Get dataframe of new events
        df_MODEL = st.session_state.gameClient.process_query(game_id, model_name=st.session_state.model) 
        
        # If there are new events: 
        if df_MODEL is not None: 
            # Make predictions on events 
            pred_MODEL = st.session_state.servingClient.predict(df_MODEL)
            
            df = pd.DataFrame(df_MODEL, columns=st.session_state.servingClient.features) 
            df = df.reset_index(drop=True)
            df['Model Output'] = pred_MODEL

            # Calculate game actual goals and goal predictions 
            real_goals, teams_A_H, teams_full = get_scores(game_id)
            pred_goals = calculate_game_goals(df_MODEL, pred_MODEL, teams_A_H) 
             
            for i in range(len(teams_A_H)):
                st.session_state.pred_goals[i] += pred_goals[i]
                st.session_state.real_goals[i] = real_goals[i] 
            st.session_state.teams = teams_A_H
            st.session_state.teams_full = teams_full

        else: 
            df = None
            st.write(':red[No new events!]')
        
        # Comparing current and previous gameId:
        if game_id == st.session_state.gameClient.gameId: 
            # st.write('Concat!')
            df = pd.concat([st.session_state.stored_df, df], ignore_index=True)
            st.session_state.stored_df = df 
        else: 
            st.session_state.stored_df = df 
        

        # Getting Game info:
        st.subheader(f"{st.session_state.teams_full[0]} VS {st.session_state.teams_full[1]}")

        period, periodTimeRemaining = get_period_info()
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



