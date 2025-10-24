import os
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

def get_data_path():
    '''Gets the absolute path to the data folder.'''
    path = os.getcwd()
    data_path = path + '/data'
    return data_path

def get_moneypuck_data(year):
    '''Gets the moneypuck data for the passed year.'''
    path_to_moneypuck_data = get_data_path() + '/moneypuck_data/moneypuck{}.csv'.format(str(year))
    moneypuck_data = pd.read_csv(path_to_moneypuck_data)
    return moneypuck_data

def get_rotowire_data(year):
    '''Gets the rotowire data for the passed year.'''
    path_to_rotowire_data = get_data_path() + '/rotowire_data/rotowire{}.csv'.format(str(year))
    rotowire_data = pd.read_csv(path_to_rotowire_data, header=1)
    return rotowire_data

