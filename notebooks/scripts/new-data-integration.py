#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Continous integraion of latest data using https://github.com/swar/nba_api/blob/master/docs/examples/Finding%20Games.ipynb


# In[1]:


#Data collection phase


# In[1]:


import pandas as pd


# In[2]:


from nba_api.stats.static import teams

nba_teams = teams.get_teams()
# Select the dictionary for the Celtics, which contains their team ID
celtics = [team for team in nba_teams if team['abbreviation'] == 'BOS'][0]
celtics_id = celtics['id']


# In[3]:


from nba_api.stats.endpoints import leaguegamefinder

# Query for games where the Celtics were playing
gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=celtics_id)
# The first DataFrame of those returned is what we want.
games = gamefinder.get_data_frames()[0]
games.head()


# In[4]:


games.groupby(games.SEASON_ID.str[-4:])[['GAME_ID']].count().loc['2015':]


# In[ ]:




