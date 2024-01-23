#!/usr/bin/env python
# coding: utf-8

# #### TODO
# - Continous integraion of latest data using https://github.com/swar/nba_api/blob/master/docs/examples/Finding%20Games.ipynb
# - Create 3 categories only that will be used for predictions.
# - Calculate TS% that will be used also for feature enginnering
# - Usage Rate (USG%): This metric estimates the percentage of team plays used by a player while he is on the floor. It can be calculated using field goal attempts, free throw attempts, and turnovers in relation to the team's total attempts.
# 
# - Assist to Turnover Ratio: A simple yet effective metric showing a player's ball-handling efficiency. It's calculated by dividing the total number of assists by the number of turnovers.
# 
# - Rebounding Efficiency: This can be calculated as the total number of rebounds (offensive + defensive) divided by the minutes played. It helps in understanding a player's rebounding ability relative to their playing time.
# 
# - Shot Selection Profile: Using FG%, 3P%, and 2P%, you can analyze a player's shot selection tendencies and efficiency.
# 
# -Add a team dimension    
# -Add relative scores based on numbers of the team.    
# -Split and extract validation data that will not be used at all, to avoid data leakage in validation   
# 
# 
# Sources :    
# https://www.kaggle.com/code/diegobormor/nba-2022-2023-data-overview/    
# https://www.kaggle.com/code/fahmisajid/player-position-classification/notebook
# 

# In[2]:


import pandas as pd
import numpy as np
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


RAW_DATA_DIR = Path('..') / "data" / "raw"
CURATED_DATA_DIR = Path('..') / "data" / "curated"


# In[4]:


filename = list(RAW_DATA_DIR.glob('*.csv'))[0]
print(f"Runnung on file: {filename}")
players = pd.read_csv(filename,sep =";", encoding='Windows-1252')


# In[5]:


players.sort_values(by=['Player'], ascending=True)


# In[6]:


assert sum(players.isnull().sum()) == 0, "There are not null values in the dataset"


# In[7]:


players["EFF"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - (players.FGA - players.FG) - (players.FTA - players.FT) - players.TOV


# In[8]:


plt.figure(figsize=(14,6))
sns.swarmplot(
    x=players["Pos"],
    y=players["EFF"]
)


# In[9]:


ages = players.Age.describe().round(decimals=1) # used to specify the first 25%, defining what is a young player
points = players.PTS.describe().round(decimals=1)


# In[10]:


sns.boxplot(data=players, y="PTS");


# In[11]:


plt.figure(figsize=(14,6))
sns.boxplot(data=players, x="Age", y="PTS");


# With the graph below, we can see that within <23y (what we have defined to be a young age), if we have more than 15 points we are special. Those data will then be used to filter the current base player and keep only special ones.

# In[12]:


young_age = ages["25%"]
futur_super_star_def = f"(EFF >= 12) & (PTS >= 15) & (Age <= {young_age})"
players.query(futur_super_star_def).sort_values("EFF", ascending=False).sort_values(["Age", "EFF"], ascending=True)


# ### Age and more

# In[13]:


sns.countplot(players['Pos'],label="Count")


# In[14]:


players


# ### True Shot percentage

# In[15]:


players['TS%'] = np.where((2 * (players['FGA'] + 0.44 * players['FTA'])) != 0, players['PTS'] / (2 * (players['FGA'] + 0.44 * players['FTA'])), 0)


# In[16]:


average_age_per_team = players.groupby('Tm')['Age'].mean()
plt.figure(figsize=(10,6))
sns.barplot(x=average_age_per_team.index, y=average_age_per_team.values)
plt.title('Average Age of Players in Each Team')
plt.xticks(rotation=90)
plt.xlabel('Team')
plt.ylabel('Average Age')
plt.show()


# In[26]:


players.Pos.value_counts()


# In[29]:


players["position"] = players.Pos.map({"PG": "Backcourt", "SG": "Backcourt", "SF": "Wing", "SF-PF": "Wing", "PF": "Big", "C": "Big", })


# In[34]:


players.position.value_counts().plot(kind='bar')


# In[ ]:





# Normalize and run a base model

# In[18]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[19]:


# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']),
        ('cat', OneHotEncoder(), ['Pos', 'Tm'])
    ])
# Define model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])


# In[20]:


# Split data into training and test sets
X = players.drop(['Player', 'PTS', 'FG%'], axis=1) # Drop target and features that are not numeric
y_pts = players['PTS'] # Points per game is the target
y_fg = players['FG%'] #  Field goals per game is the target
X_train_pts, X_test_pts, y_train_pts, y_test_pts = train_test_split(X, y_pts, test_size=0.2, random_state=42)
X_train_fg, X_test_fg, y_train_fg, y_test_fg = train_test_split(X, y_fg, test_size=0.2, random_state=42)


# In[21]:


# Train model to predict PTS
model.fit(X_train_pts, y_train_pts)
pts_preds = model.predict(X_test_pts)
print(f'RMSE for PTS prediction: {mean_squared_error(y_test_pts, pts_preds, squared=False)}')


# In[22]:


# Train model to predict FG%
model.fit(X_train_fg, y_train_fg)
fg_preds = model.predict(X_test_fg)
print(f'RMSE for FG% prediction: {mean_squared_error(y_test_fg, fg_preds, squared=False)}')


# In[ ]:




