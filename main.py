import os
import numpy as np
import pandas as pd
from player import *
def driver():
    #Changes the working directory to the main repository directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #Puts player data csv into "df" dataframe variable
    path = os.getcwd()
    data_path = path + '\\data'
    df = pd.read_csv(data_path + '\\rotowire_data\\rotowire2021.csv')
    #df.replace() = "PP_Goals"
    
    rw_labels = ["Player Name", "Team", "Pos", "Games", "Goals", "Assists", "Pts", "+/-", "PIM", "SOG", "GWG", "PP_Goals", "PP_Assists", "SH_Goals", "SH_Assists", "Hits", "Blocked_Shots"]
    #df.iloc[0] = rw_labels
    df.set_axis(rw_labels, axis=1, inplace=True)
    df.drop(index=df.index[0], axis=0, inplace=True)
    
    fantasy_points = [0 for i in range(len(df))]
    print(fantasy_points)
    
    for label in rw_labels:
        print(label)
        if label == "Goals":
            fantasy_points = np.add(fantasy_points, [5*int(num) for num in df.loc[:, label]])
        if label == "Assists":
            fantasy_points = np.add(fantasy_points, [3*int(num) for num in df.loc[:, label]])
        if label == "+/-":
            fantasy_points = np.add(fantasy_points, [1.5*int(num) for num in df.loc[:, label]])
        if label == "PIM":
            fantasy_points = np.add(fantasy_points, [-.25*int(num) for num in df.loc[:, label]])
        if label == "PP_Goals":
            fantasy_points = np.add(fantasy_points, [4*int(num) for num in df.loc[:, label]])
        if label == "PP_Assists":
            fantasy_points = np.add(fantasy_points, [2*int(num) for num in df.loc[:, label]])
        if label == "SH_Goals":
            fantasy_points = np.add(fantasy_points, [6*int(num) for num in df.loc[:, label]])
        if label == "SH_Assists":
            fantasy_points = np.add(fantasy_points, [4*int(num) for num in df.loc[:, label]])
        if label == "+/-":
            fantasy_points = np.add(fantasy_points, [5*int(num) for num in df.loc[:, label]])
        
    print(fantasy_points)
    print(df.loc[:, "Goals"])
driver()