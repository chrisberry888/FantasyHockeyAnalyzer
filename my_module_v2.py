import os
import pandas as pd
from pandas import DataFrame
import re
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import requests

def replace_team_abbreviations(df: DataFrame) -> DataFrame:
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


def get_player_id_table(yearly_player_data):
    '''Gets a table that translates playerIds to names/positions/teams'''
    all_years = pd.concat(yearly_player_data)
    all_years = all_years[['playerId', 'season', 'name', 'team', 'position']]

    # Sorts by season (highest first), then keeps the first row for each playerId
    final_df = all_years.sort_values('season', ascending=False).drop_duplicates(subset='playerId')

    final_df = pd.DataFrame({
        'playerId': final_df['playerId'],
        'name': final_df['name'].astype(str) + '_' + final_df['team'].astype(str) + '_' + final_df['season'].astype(str) + '_' + final_df['position'].astype(str)
    })
    return final_df


def pivot_data(df: DataFrame) -> DataFrame:
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


def preformat_moneypuck_df(df: DataFrame) -> DataFrame:
    '''Prepares the moneypuck dataframe for merging.'''
    # df = rename_situation_column(df)
    df = pivot_data(df)
    return df
    

def get_rid_of_irrelevant_rotowire_columns(df: DataFrame) -> DataFrame:
    '''Returns a new dataframe, only with the columns we want from the rotowire dataframe'''
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


def rename_rotowire_columns(df: DataFrame) -> DataFrame:
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


def preformat_rotowire_df(df: DataFrame) -> DataFrame:
    '''Prepares the rotowire dataframe for merging.'''
    df = get_rid_of_irrelevant_rotowire_columns(df)
    df = rename_rotowire_columns(df)
    return df


def merge_dataframes(moneypuck_df, rotowire_df):
    '''Merges the two dataframes (after they've already been preformatted).'''
    primary_keys = ['name', 'games_played', 'all_I_F_shotsOnGoal', 'all_I_F_hits']
    merged_df = pd.merge(moneypuck_df, rotowire_df, on=primary_keys)
    merged_df = replace_team_abbreviations(merged_df)
    return merged_df


def format_fantasy_columns(df: DataFrame) -> DataFrame:
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


def merge_dataframes_for_ml(list_of_dataframes, points_df=None):
    '''
    Merges the dataframes in list_of_dataframes, then tacks on the Fantasy_Points column from points_df.
    If points_df isn't passed in, then it's the dataframe for the most recent year.
    '''
    list_of_dataframes = [df.drop(columns=['Fantasy_Points']) for df in list_of_dataframes]
    final_df = list_of_dataframes[0]
    for index, df in enumerate(list_of_dataframes):
        if index == 0:
            continue
        final_df = pd.merge(final_df, df, on='playerId', suffixes=(None, f'_{index}'))

    if points_df is not None:
        points_df = points_df[['playerId', 'Fantasy_Points']]
        final_df = pd.merge(final_df, points_df, on='playerId')
    return final_df

    
def encode_data(df: DataFrame) -> DataFrame:
    '''Encodes string data as floats (and drops irrelevant columns).'''
    df = df.drop(columns=['name'])
    df = pd.get_dummies(df, columns=['team', 'position'])
    return df


def ml_data_post_processing(df: DataFrame) -> DataFrame:
    '''Replaces any NaN's with False, and sorts the columns alphabetically to make sklearn not angry'''
    df = df.fillna(False)
    sorted_column_names = sorted(df.columns)
    df = df[sorted_column_names]
    return df


# TODO: Clean up this function, make it more readable/make it more clear for the reader
def get_ml_data(yearly_player_data, current_year, number_of_years_per_row):
    '''Gets the dataframes that will be used by the ML model trainers.'''
    final_df = pd.DataFrame()
    first_year_with_data = 2010
    last_year_with_data = current_year - number_of_years_per_row
    for year in range(first_year_with_data, last_year_with_data):
        first_index = year - first_year_with_data
        last_index = first_index + number_of_years_per_row
        relevant_dfs = [encode_data(yearly_player_data[i]) for i in range(first_index, last_index)]
        ml_data = merge_dataframes_for_ml(relevant_dfs, yearly_player_data[first_index + number_of_years_per_row])
        
        #does pd.concat do what you want?
        final_df = pd.concat([final_df, ml_data], ignore_index=True)
    final_df = ml_data_post_processing(final_df)
    return final_df


def separate_fantasy_points(df):
    fantasy_points = df['Fantasy_Points'].tolist()
    df = df.drop(columns=['Fantasy_Points'])
    return (df, fantasy_points)


def set_up_directory(base_path: str, subdirectory_name: str):
    target_directory = Path(base_path) / subdirectory_name
    if target_directory.exists():
        # Delete all files
        for item in target_directory.iterdir():
            if item.is_file():
                item.unlink() # 'unlink' is the pathlib method to delete a file
    else:
        target_directory.mkdir(parents=True)


def create_models(X, y, blank_model, number_of_models, year_path_name, model_path_name):
    '''Creates the ML models and dumps them into a file in the /models directory'''
    base_path = os.getcwd()
    set_up_directory(f'{base_path}/models/{year_path_name}', model_path_name)
    full_folder_path = f'{base_path}/models/{year_path_name}/{model_path_name}'
    for i in range(number_of_models):
        current_model = clone(blank_model)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        current_model.fit(X_train, y_train)
        file_name = f'{full_folder_path}/model_{i}.joblib'
        joblib.dump(current_model, file_name)


def get_final_year_data(yearly_player_data, number_of_years):
    '''Returns data that can be used by the already-created models'''
    first_index = number_of_years * -1
    relevant_dfs = [encode_data(yearly_player_data[i]) for i in range(first_index, 0)]
    final_df = merge_dataframes_for_ml(relevant_dfs)
    final_df = ml_data_post_processing(final_df)
    return final_df
    

def get_prediction_table(models, input_data, player_id_table):
    prediction_values = [0 for player in range(len(input_data))]
    for model in models:
        current_prediction = model.predict(input_data)
        prediction_values = [x + y for x, y in zip(prediction_values, current_prediction)]
    preds_df = pd.DataFrame({
        'playerId': input_data['playerId'].values,
        'prediction': prediction_values
    })
    final_df = pd.merge(player_id_table, preds_df, on='playerId').sort_values(by='prediction', ascending=False).reset_index(drop=True)
    return final_df


def get_formatted_prediction_table(prediction, input_table, player_id_table):
    '''Takes the raw predicion table as input with the player_id_table; returns the final useful table for this one model'''
    preds_df = pd.DataFrame({
        'playerId': input_table['playerId'].values,
        'prediction': prediction
    })
    df = pd.merge(player_id_table, preds_df, on='playerId').sort_values(by='prediction', ascending=False).reset_index(drop=True)
    return df


def generate_predictions(year_data: tuple, player_id_table):
    '''Iterates through all of the model files and generates predictions for all of them.'''
    current_working_directory = os.getcwd()
    lowest_directory_pattern = r'models/(1|2|3)_year/(neural_nets|random_forests|boosted_trees)'
    for directory, subdirectories, files in os.walk(f'{current_working_directory}/models'):
        match = re.search(lowest_directory_pattern, directory)
        if match:
            correct_input = year_data[int(match.group(1)) - 1]
            # Deletes all predictions that are currently in the predictions folder for this 
            set_up_directory(f'{current_working_directory}/predictions/{match.group(1)}_year', match.group(2))
            
            for i, model_name in enumerate(files):
                current_model = joblib.load(f'{directory}/{model_name}')
                current_prediction = current_model.predict(correct_input)
                prediction_table = get_formatted_prediction_table(current_prediction, correct_input, player_id_table)
                table_location = f'{current_working_directory}/predictions/{match.group(1)}_year/{match.group(2)}'
                prediction_table.to_parquet(f'{table_location}/prediction_{i}.parquet')


def generate_final_table():
    '''Averages each player's available predictions and reports model coverage as a percentage.'''
    current_working_directory = Path(os.getcwd())
    prediction_paths = sorted((current_working_directory / 'predictions').rglob('*.parquet'))
    if not prediction_paths:
        raise ValueError('No prediction parquet files found.')

    prediction_tables = []
    for model_id, current_path in enumerate(prediction_paths):
        current_df = pd.read_parquet(current_path)
        # Give each model one contribution per player, even if a file repeats a player.
        current_df = current_df.groupby(['playerId', 'name'], as_index=False)['prediction'].mean()
        current_df = current_df.dropna(subset=['prediction'])
        current_df['_model_id'] = model_id
        prediction_tables.append(current_df)

    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    final_df = all_predictions.groupby(['playerId', 'name'], as_index=False).agg(
        prediction=('prediction', 'mean'),
        _model_count=('_model_id', 'nunique')
    )
    final_df['model_coverage_pct'] = (100 * final_df['_model_count'] / len(prediction_paths)).round(1)
    final_df = final_df.drop(columns=['_model_count'])
    final_df = final_df.sort_values(by='prediction', ascending=False).reset_index(drop=True)
    final_df.to_csv(current_working_directory / 'final_prediction.csv', index=False)


def get_final_table():
    current_working_directory = os.getcwd()
    loaded_df = pd.read_csv(f'{current_working_directory}/final_prediction.csv')
    return loaded_df



def get_nhl_players() -> pd.DataFrame:
    base = "https://api.nhle.com/stats/rest/en"
    teams = requests.get(f"{base}/team", timeout=30).json()["data"]

    rows = []
    for team in teams:
        players = requests.get(
            f"{base}/players",
            params={"limit": 1000, "cayenneExp": f"currentTeamId={team['id']}"},
            timeout=30,
        ).json()["data"]
        rows += [
            {
                "player_id": p["id"],
                "team_abbrev": team["triCode"],
                "position": p["positionCode"],
            }
            for p in players
        ]

    return pd.DataFrame(rows, columns=["player_id", "team_abbrev", "position"])