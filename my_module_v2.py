import os
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

def replace_team_abbreviations(df):
    '''Replaces the team abbreviation to the most updated one.'''
    abbreviation_table = {
        'ANH': 'ANA',
        'ARI': 'UTA',
        'ATL': 'WPG',
        'CLS': 'CBJ',
        'L.A': 'LAK',
        'LA': 'LAK',
        'MON': 'MTL',
        'N.J': 'NJD',
        'NJ': 'NJD',
        'S.J': 'SJS',
        'SJ': 'SJS',
        'T.B': 'TBL',
        'TB': 'TBL',
        'WAS': 'WSH'
    }
    df['team'] = df['team'].replace(abbreviation_table)
    return df


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

def rename_situation_column(df):
    '''Renames the elements in the situation column in the moneypuck data.'''
    rename_dict = {
        'other': 'other',
        'all': 'all',
        '5on5': 'full_strength',
        '4on5': 'SH',
        '5on4': 'PP'
    }
    df['situation'] = df['situation'].replace(rename_dict)
    return df

def pivot_data(df):
    '''Puts the data from the different situations (5on5, 5on4, etc) into the same row.'''
    key_columns = [
        'playerId',
        'season',
        'name',
        'team',
        'position',
        'games_played'
    ]
    pivoted_df = df.pivot(index=key_columns, columns='situation')
    pivoted_df.columns = [f"{sit}_{col}" for col, sit in pivoted_df.columns]
    final_df = pivoted_df.reset_index()
    return final_df


def preformat_moneypuck_df(df):
    '''Prepares the moneypuck dataframe for merging.'''
    # df = rename_situation_column(df)
    df = pivot_data(df)
    return df
    

def get_rid_of_irrelevant_rotowire_columns(df):
    filter_columns = [
        'Player Name', 
        'Games', 
        'SOG', 
        'Hits',
        '+/-',
        'A',
        'G.1', #power play/short handed stats are imported like this
        'A.1',
        'G.2',
        'A.2'
    ]
    df = df[filter_columns]
    return df


def rename_rotowire_columns(df):
    rename_dict_1 = {
        'A': 'Assists',
        'G.1': 'PP_Goals',
        'A.1': 'PP_Assists',
        'G.2': 'SH_Goals',
        'A.2': 'SH_Assists'
    }
    moneypuck_columns = ['name', 'games_played', 'all_I_F_shotsOnGoal', 'all_I_F_hits']
    rotowire_columns = ['Player Name', 'Games', 'SOG', 'Hits']
    rename_dict_2 = dict(zip(rotowire_columns, moneypuck_columns))
    df = df.rename(columns=rename_dict_1)
    df = df.rename(columns=rename_dict_2)
    return df


def preformat_rotowire_df(df):
    '''Prepares the rotowire dataframe for merging.'''
    df = get_rid_of_irrelevant_rotowire_columns(df)
    df = rename_rotowire_columns(df)
    return df


def merge_dataframes(moneypuck_df, rotowire_df):
    '''Merges the two dataframes (after they've already been preformatted).'''
    primary_keys = ['name', 'games_played', 'all_I_F_shotsOnGoal', 'all_I_F_hits']
    merged_df = pd.merge(moneypuck_df, rotowire_df, on=primary_keys)
    return merged_df


def format_fantasy_columns(df):
    '''Adds all columns needed to calculate fantasy points'''
    rename_dict = {
        'all_I_F_goals': 'Goals',
        'Assists': 'Assists',
        '+/-': '+/-',
        'all_I_F_penalityMinutes': 'PIM',
        'PP_Goals': 'PP_Goals',
        'PP_Assists': 'PP_Assists',
        'SH_Goals': 'SH_Goals',
        'SH_Assists': 'SH_Assists',
        'all_faceoffsWon': 'Faceoffs_Won',
        'all_faceoffsLost': 'Faceoffs_Lost',
        'all_I_F_hits': 'Hits',
        'all_shotsBlockedByPlayer': 'Blocked_Shots'
    }
    df = df.rename(columns=rename_dict)
    return df


def combine_dataframes(moneypuck_df, rotowire_df):
    '''Formats then combines the moneypuck and rotowire dataframes.'''
    moneypuck_df = preformat_moneypuck_df(moneypuck_df)
    rotowire_df = preformat_rotowire_df(rotowire_df)
    merged_df = merge_dataframes(moneypuck_df, rotowire_df)
    merged_df = format_fantasy_columns(merged_df)
    return merged_df
    

def calculate_fantasy_points(df, points_dictionary):
    '''Adds a column with each player's fantasy output for that year'''
    df['Fantasy_Points'] = sum(df[col] * multiplier for col, multiplier in points_dictionary.items())
    return df

