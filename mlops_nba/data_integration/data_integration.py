from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

def fetch_nba_team_data(team_abbreviation):
    nba_teams = teams.get_teams()
    team = [team for team in nba_teams if team['abbreviation'] == team_abbreviation][0]
    team_id = team['id']
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    return games

def integrate_data(*dataframes):
    return pd.concat(dataframes, ignore_index=True)
