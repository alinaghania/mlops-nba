import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

RAW_DATA_DIR = Path('..') / "data" / "raw"
CURATED_DATA_DIR = Path('..') / "data" / "curated"

def load_data():
    filename = list(RAW_DATA_DIR.glob('*.csv'))[0]
    print(f"Running on file: {filename}")
    players = pd.read_csv(filename, sep=";", encoding='Windows-1252')
    return players

def check_null_values(players):
    assert sum(players.isnull().sum()) == 0, "There are null values in the dataset"
    return players

def calculate_efficiency(players):
    players["EFF"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - \
                     (players.FGA - players.FG) - (players.FTA - players.FT) - players.TOV
    return players

def calculate_true_shooting_percentage(players):
    players['TS%'] = np.where((2 * (players['FGA'] + 0.44 * players['FTA'])) != 0, 
                              players['PTS'] / (2 * (players['FGA'] + 0.44 * players['FTA'])), 0)
    return players

def map_positions(players):
    players["position"] = players.Pos.map({"PG": "Backcourt", "SG": "Backcourt",
                                           "SF": "Wing", "SF-PF": "Wing",
                                           "PF": "Big", "C": "Big", })
    return players

def process_data():
    players = load_data()
    players = check_null_values(players)
    players = calculate_efficiency(players)
    players = calculate_true_shooting_percentage(players)
    players = map_positions(players)
    
    return players

if __name__ == "__main__":
    processed_data = process_eda_data()
    processed_data.to_csv(CURATED_DATA_DIR / 'processed_players.csv', index=False)
