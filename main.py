import os
import pandas as pd
from player import *
def driver():
    #Changes the working directory to the main repository directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #Puts player data csv into "df" dataframe variable
    path = os.getcwd()
    data_path = path + '\\data'
    df = pd.read_csv(data_path + '\skaters21-22.csv')
    print(df["I_F_goals"])


driver()