import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import requests
import json


def distance_shot(x: float, y: float):
    """
    Calculer la distance entre le tir et le filet

    Parameters
    ----------
    x: float
        Coordonnée en x du tir
    y: float
        Coordonnée en y du tir
    
    Returns
    -------
    distance: array
        Distance entre le tir et le filet pour tous les tirs
    """
    # Les coordonnées du filet ont été définies comme le centre de la ligne des buts.
    # Sachant que la ligne des buts est située à 11 pieds de la bande et que la patinoire
    # a une largeur de 200 pieds, nous pouvons calculer la coordonnées en x en faisant
    # 200/2 - 11 = 89. La coordonnée en y est simplement 0.
    x_goal, y_goal = 89, 0

    # Calculer la distance entre le tir et le filet
    distance = np.sqrt((x_goal - np.abs(x))**2 + (y_goal - np.abs(y))**2)

    return distance


def angle_shot(x: float, y: float):
    """
    Calculer l'angle entre le tir et le filet

    Parameters
    ----------
    x: float
        Coordonnée en x du tir
    y: float
        Coordonnée en y du tir

    Returns
    -------
    angle: array
        Angle entre le tir et le filet pour tous les tirs
    """
    # Les coordonnées du filet ont été définies comme le centre de la ligne des buts.
    # Sachant que la ligne des buts est située à 11 pieds de la bande et que la patinoire
    # a une largeur de 200 pieds, nous pouvons calculer la coordonnées en x en faisant
    # 200/2 - 11 = 89. La coordonnée en y est simplement 0.
    x_goal, y_goal = 89, 0

    # Calculer l'angle entre le tir et le filet
    angle = np.arctan((y_goal - y)/(x_goal - np.abs(x)))
    # Convertir l'angle en degrés
    angle = np.rad2deg(angle)

    return angle

def goaliePresent(data, homeTeam_id):
    """
    Détermine si le filet était vide ou non en se basant sur le `situationCode` et `eventOwnerTeamId`.

    Parameters
    ----------
    data : DataFrame
        DataFrame contenant les données de jeu, y compris `situationCode` et `eventOwnerTeamId`.

    Returns
    -------
    empty_goal : array
        Variable binaire indiquant si le filet était désert ou non
    """
    def is_empty_net(situation_code, event_owner_team_id):
        # S'assurer que situation_code a au moins 4 caractères
        if len(situation_code) < 4:
            return 0  # ou une autre valeur par défaut appropriée

        # Décomposition du situationCode
        away_goalie_present = int(situation_code[0])
        home_goalie_present = int(situation_code[3])

        # Déterminer si le filet est vide
        if event_owner_team_id == homeTeam_id:
            return 1 if home_goalie_present == 1 else 0
        else:
            return 1 if away_goalie_present == 1 else 0

    # Appliquer la fonction à chaque ligne
    data['emptyNet'] = data.apply(lambda row: is_empty_net(row['situationCode'], row['eventOwnerTeamId']), axis=1)

    # Encoder en binaire
    empty_goal = LabelEncoder().fit_transform(data['emptyNet'])

    return empty_goal




def retrieve_nhl_play_by_play_modified(game_id):
    api_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    response = requests.get(api_url)
    if response.status_code != 200:
        return f"Erreur lors de la récupération des données: Code de statut {response.status_code}"

    data = response.json()
    
    awayTeam_id = data['awayTeam']['id']
    homeTeam_id = data['homeTeam']['id']
    
    extracted_data = []
    for event in data['plays']:
        details = event.get('details', {})
        x_coord = details.get('xCoord')
        y_coord = details.get('yCoord')
        if event.get('typeDescKey') == 'shot-on-goal':
            is_goal_flag = 1
        else:
            is_goal_flag = 0
        situation_code = event.get('situationCode', '')
        event_owner_team_id = details.get('eventOwnerTeamId')

        extracted_data.append({
            "eventId": event['eventId'],
            "xCoord": x_coord,
            "yCoord": y_coord,
            "isGoal": is_goal_flag,
            "situationCode": situation_code,
            "eventOwnerTeamId": event_owner_team_id
        })

    # Convertir en DataFrame
    df_extracted_data = pd.DataFrame(extracted_data)
    df = pd.DataFrame(columns=['shot_distance', 'shot_angle', 'empty_goal','is_goal'])

    # Appliquer les fonctions
    df['shot_distance'] = df_extracted_data.apply(lambda row: distance_shot(row['xCoord'], row['yCoord']), axis=1)
    df['shot_angle'] = df_extracted_data.apply(lambda row: angle_shot(row['xCoord'], row['yCoord']), axis=1)
    df['empty_goal'] = goaliePresent(df_extracted_data, awayTeam_id)
    df['is_goal'] = df_extracted_data['isGoal']

    # Gestion des valeurs manquantes
    df['shot_distance'].fillna(0, inplace=True)
    df['is_goal'].fillna(0, inplace=True)
    df['empty_goal'].fillna(0, inplace=True)
    df['shot_angle'].fillna(0, inplace=True)

    return df