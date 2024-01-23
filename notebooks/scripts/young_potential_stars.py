#!/usr/bin/env python
# coding: utf-8

# 
# ## Introduction
# 
# In this analysis, we will be calculating the young stars who have the best potential. We will use current data to define the age of a young player, their efficiency based on others, and other criteria. Our goal is to identify the players with the highest potential and analyze their performance in various aspects.
# 
# Let's get started!
# 
# 
# Source:   
# This work was highly inspired by this work: https://www.kaggle.com/code/vivovinco/nba-rising-stars-2022-2023

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


# %%
