import os
import errno
import json
import shutil
import tempfile
import pandas as pd
from pandas import DataFrame
import re
import unicodedata
from sklearn.base import clone
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
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
        'Team',
        'Pos',
        '+/-',
        'A',
        'G.1', #power play/short handed stats are imported like this
        'A.1',
        'G.2',
        'A.2',
        'GWG'
    ]
    df = df[filter_columns]
    return df


def rename_rotowire_columns(df: DataFrame) -> DataFrame:
    rename_dict_1 = {
        'A': 'Assists',
        'G.1': 'PP_Goals',
        'A.1': 'PP_Assists',
        'G.2': 'SH_Goals',
        'A.2': 'SH_Assists',
        'GWG': 'Game_Winning_Goals'
    }
    moneypuck_columns = ['name', 'team', 'position']
    rotowire_columns = ['Player Name', 'Team', 'Pos']
    rename_dict_2 = dict(zip(rotowire_columns, moneypuck_columns))
    df = df.rename(columns=rename_dict_1)
    df = df.rename(columns=rename_dict_2)
    return df


def preformat_rotowire_df(df: DataFrame) -> DataFrame:
    '''Prepares the rotowire dataframe for merging.'''
    df = get_rid_of_irrelevant_rotowire_columns(df)
    df = rename_rotowire_columns(df)
    return df


def _normalize_player_name(name):
    if pd.isna(name):
        return ''
    normalized = unicodedata.normalize('NFKD', str(name).casefold())
    return re.sub(r'[^a-z0-9]', '', normalized)


def _get_position_group(position):
    position = str(position).strip().upper()
    if position in {'C', 'LW', 'RW', 'L', 'R', 'F'}:
        return 'F'
    return position if position in {'D', 'G'} else ''


def merge_dataframes(moneypuck_df, rotowire_df, return_unmatched=False):
    '''Matches annual player identities, then merges on MoneyPuck player IDs.'''
    moneypuck_df = moneypuck_df.reset_index(drop=True)
    rotowire_df = rotowire_df.reset_index(drop=True)
    alias_path = Path(__file__).resolve().parent / 'data' / 'player_name_aliases.csv'
    alias_table = pd.read_csv(alias_path)
    aliases = dict(zip(
        alias_table['alias'].map(_normalize_player_name),
        alias_table['canonical_name'].map(_normalize_player_name)
    ))

    mp_identity = moneypuck_df[['playerId', 'name', 'position']].copy()
    rw_identity = rotowire_df[['name', 'position']].copy()
    rw_identity['_rw_row_id'] = range(len(rw_identity))
    for identity in [mp_identity, rw_identity]:
        names = identity['name'].map(_normalize_player_name)
        identity['_name_key'] = names.map(lambda name: aliases.get(name, name))
        identity['_position_group'] = identity['position'].map(_get_position_group)

    mp_name_counts = mp_identity.groupby('_name_key')['playerId'].transform('size')
    rw_name_counts = rw_identity.groupby('_name_key')['_rw_row_id'].transform('size')
    unique_mp = mp_identity.loc[mp_name_counts.eq(1) & mp_identity['_name_key'].ne('')]
    unique_rw = rw_identity.loc[rw_name_counts.eq(1) & rw_identity['_name_key'].ne('')]
    matches = unique_mp[['playerId', '_name_key']].merge(
        unique_rw[['_rw_row_id', '_name_key']], on='_name_key', validate='one_to_one'
    )[['playerId', '_rw_row_id']]

    # Resolve repeated names only when the position group uniquely identifies both rows.
    remaining_mp = mp_identity.loc[~mp_identity['playerId'].isin(matches['playerId'])]
    remaining_rw = rw_identity.loc[~rw_identity['_rw_row_id'].isin(matches['_rw_row_id'])]
    keys = ['_name_key', '_position_group']
    mp_group_counts = remaining_mp.groupby(keys)['playerId'].transform('size')
    rw_group_counts = remaining_rw.groupby(keys)['_rw_row_id'].transform('size')
    unique_mp = remaining_mp.loc[
        mp_group_counts.eq(1) & remaining_mp['_name_key'].ne('') & remaining_mp['_position_group'].ne('')
    ]
    unique_rw = remaining_rw.loc[
        rw_group_counts.eq(1) & remaining_rw['_name_key'].ne('') & remaining_rw['_position_group'].ne('')
    ]
    position_matches = unique_mp[['playerId'] + keys].merge(
        unique_rw[['_rw_row_id'] + keys], on=keys, validate='one_to_one'
    )[['playerId', '_rw_row_id']]
    matches = pd.concat([matches, position_matches], ignore_index=True)

    rw_with_ids = matches.merge(
        rotowire_df.assign(_rw_row_id=range(len(rotowire_df))),
        on='_rw_row_id', validate='one_to_one'
    )
    fantasy_columns = ['+/-', 'Assists', 'PP_Goals', 'PP_Assists', 'SH_Goals', 'SH_Assists', 'Game_Winning_Goals']
    merged_df = moneypuck_df.merge(
        rw_with_ids[['playerId'] + fantasy_columns], on='playerId', validate='one_to_one'
    )
    merged_df = replace_team_abbreviations(merged_df)

    report_columns = ['season', 'source', 'playerId', 'name', 'team', 'position', 'reason']
    mp_unmatched = moneypuck_df.loc[
        ~moneypuck_df['playerId'].isin(matches['playerId']),
        ['season', 'playerId', 'name', 'team', 'position']
    ].copy()
    mp_unmatched['source'] = 'moneypuck'
    mp_unmatched['reason'] = mp_identity.loc[mp_unmatched.index, '_name_key'].isin(
        rw_identity['_name_key']
    ).map({True: 'ambiguous_name_or_position', False: 'no_matching_name'})

    rw_unmatched = rotowire_df.loc[
        ~rw_identity['_rw_row_id'].isin(matches['_rw_row_id']), ['name', 'team', 'position']
    ].copy()
    rw_unmatched['season'] = moneypuck_df['season'].iloc[0] if not moneypuck_df.empty else pd.NA
    rw_unmatched['source'] = 'rotowire'
    rw_unmatched['playerId'] = pd.NA
    rw_unmatched['reason'] = rw_identity.loc[rw_unmatched.index, '_name_key'].isin(
        mp_identity['_name_key']
    ).map({True: 'ambiguous_name_or_position', False: 'no_matching_name'})
    unmatched_df = pd.concat([
        mp_unmatched.reindex(columns=report_columns), rw_unmatched.reindex(columns=report_columns)
    ], ignore_index=True)
    unmatched_df['playerId'] = unmatched_df['playerId'].astype('Int64')
    return (merged_df, unmatched_df) if return_unmatched else merged_df


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


def combine_dataframes(moneypuck_df, rotowire_df, return_unmatched=False):
    '''Formats and combines annual data, optionally returning unmatched player identities.'''
    moneypuck_df = preformat_moneypuck_df(moneypuck_df)
    rotowire_df = preformat_rotowire_df(rotowire_df)
    merged_df, unmatched_df = merge_dataframes(moneypuck_df, rotowire_df, return_unmatched=True)
    merged_df = format_fantasy_columns(merged_df)
    return (merged_df, unmatched_df) if return_unmatched else merged_df
    

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
    '''Separates targets and removes player IDs from model features.'''
    fantasy_points = df['Fantasy_Points'].tolist()
    df = df.drop(columns=['Fantasy_Points', 'playerId'])
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


def reset_model_and_prediction_directories():
    '''Clears all contents of the project's models and predictions directories.'''
    directories = [Path(os.getcwd()) / name for name in ('models', 'predictions')]
    for directory in directories:
        if directory.is_symlink():
            raise ValueError(f'Refusing to reset a linked output directory: {directory}')
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(directory)

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        for item in directory.iterdir():
            if item.is_symlink() or not item.is_dir():
                item.unlink()
            else:
                shutil.rmtree(item)


def _atomic_write(path, write_contents):
    '''Flushes a temporary file to disk before publishing its final filename.'''
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
    )
    try:
        with os.fdopen(descriptor, 'wb') as output:
            write_contents(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        # Persist the rename where the filesystem supports syncing directories.
        if hasattr(os, 'O_DIRECTORY'):
            directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                try:
                    os.fsync(directory_descriptor)
                except OSError as error:
                    if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                        raise
            finally:
                os.close(directory_descriptor)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _training_metadata(X, y, blank_model, number_of_models):
    '''Identifies the data, feature layout, and estimator used by one model group.'''
    # Compare values rather than pandas' internal memory layout across restarts.
    row_hashes = pd.util.hash_pandas_object(X, index=True).to_numpy()
    return {
        'format_version': 1,
        'training_data_hash': joblib.hash((row_hashes, y)),
        'feature_names': list(X.columns),
        'feature_dtypes': [str(dtype) for dtype in X.dtypes],
        'estimator_hash': joblib.hash(clone(blank_model)),
        'estimator': repr(blank_model),
        'number_of_models': number_of_models,
        'versions': {
            'sklearn': sklearn_version,
            'pandas': pd.__version__,
            'joblib': joblib.__version__,
        },
    }


def _completed_model_matches(path, metadata):
    '''Checks that a saved model is readable, fitted, and belongs to this run.'''
    try:
        model = joblib.load(path)
        check_is_fitted(model)
        return (
            joblib.hash(clone(model)) == metadata['estimator_hash']
            and model.feature_names_in_.tolist() == metadata['feature_names']
        )
    except Exception:
        # Interrupted or damaged model files are retrained rather than reused.
        return False


def create_models(X, y, blank_model, number_of_models, year_path_name, model_path_name,
                  resume_training=True):
    '''Saves models individually, optionally resuming an interrupted matching run.'''
    fresh_start_hint = (
        'Set create_new_models=True in the notebook for a full reset. '
        'For direct calls, use resume_training=False to start this model group fresh.'
    )
    full_folder_path = Path(os.getcwd()) / 'models' / year_path_name / model_path_name
    metadata = _training_metadata(X, y, blank_model, number_of_models)
    metadata_path = full_folder_path / 'training_metadata.json'
    if resume_training:
        if metadata_path.exists():
            try:
                saved_metadata = json.loads(metadata_path.read_text())
            except (OSError, ValueError) as error:
                raise ValueError(
                    f'Cannot verify training metadata in {full_folder_path}. '
                    + fresh_start_hint
                ) from error
            if saved_metadata != metadata:
                raise ValueError(
                    f'Training data, features, settings, or versions changed for '
                    f'{year_path_name}/{model_path_name}. '
                    + fresh_start_hint
                )
        elif full_folder_path.exists() and any(full_folder_path.glob('model_*.joblib')):
            raise ValueError(
                f'Saved models in {full_folder_path} have no training metadata. '
                + fresh_start_hint
            )
        full_folder_path.mkdir(parents=True, exist_ok=True)
    else:
        # Preserve the existing fresh-start behavior for this model group only.
        set_up_directory(full_folder_path.parent, model_path_name)
    if not resume_training or not metadata_path.exists():
        contents = (json.dumps(metadata, indent=2) + '\n').encode('utf-8')
        _atomic_write(metadata_path, lambda output: output.write(contents))

    reused_models = 0
    for i in range(number_of_models):
        file_name = full_folder_path / f'model_{i}.joblib'
        if resume_training and file_name.exists() and _completed_model_matches(file_name, metadata):
            reused_models += 1
            continue
        current_model = clone(blank_model)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        current_model.fit(X_train, y_train)
        _atomic_write(file_name, lambda output: joblib.dump(current_model, output))
        print(f'{year_path_name}/{model_path_name}: saved model {i + 1}/{number_of_models}',
              flush=True)
    print(f'{year_path_name}/{model_path_name}: reused {reused_models}, '
          f'trained {number_of_models - reused_models}, complete {number_of_models}', flush=True)


def get_final_year_data(yearly_player_data, number_of_years):
    '''Returns prediction data, including player IDs for identifying results.'''
    first_index = number_of_years * -1
    relevant_dfs = [encode_data(yearly_player_data[i]) for i in range(first_index, 0)]
    final_df = merge_dataframes_for_ml(relevant_dfs)
    final_df = ml_data_post_processing(final_df)
    return final_df
    

def get_prediction_table(models, input_data, player_id_table):
    model_features = input_data.drop(columns=['playerId'])
    prediction_values = [0 for player in range(len(input_data))]
    for model in models:
        current_prediction = model.predict(model_features)
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
            model_features = correct_input.drop(columns=['playerId'])
            # Deletes all predictions that are currently in the predictions folder for this 
            set_up_directory(f'{current_working_directory}/predictions/{match.group(1)}_year', match.group(2))
            
            model_names = sorted(
                (name for name in files if re.fullmatch(r'model_\d+\.joblib', name)),
                key=lambda name: int(Path(name).stem.split('_')[-1]),
            )
            for i, model_name in enumerate(model_names):
                current_model = joblib.load(f'{directory}/{model_name}')
                current_prediction = current_model.predict(model_features)
                prediction_table = get_formatted_prediction_table(current_prediction, correct_input, player_id_table)
                table_location = f'{current_working_directory}/predictions/{match.group(1)}_year/{match.group(2)}'
                prediction_table.to_parquet(f'{table_location}/prediction_{i}.parquet')


def generate_final_table():
    '''Averages available predictions and adds team and position from the NHL API.'''
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
    nhl_players = get_nhl_players().rename(columns={
        'player_id': 'playerId', 'team_abbrev': 'Team', 'position': 'Position'
    })
    final_df = final_df.merge(
        nhl_players[['playerId', 'Team', 'Position']],
        on='playerId', how='left', validate='many_to_one'
    )
    final_df[['Team', 'Position']] = final_df[['Team', 'Position']].fillna('UNKNOWN')
    final_df['name'] = final_df['name'].str.rsplit('_', n=3).str[0]
    final_df = final_df.rename(columns={
        'playerId': 'playerID', 'name': 'Player Name', 'prediction': 'Prediction',
        'model_coverage_pct': 'model_coverage_percent'
    })
    final_df = final_df[[
        'playerID', 'Player Name', 'Team', 'Position', 'Prediction', 'model_coverage_percent'
    ]]
    final_df.to_csv(current_working_directory / 'final_prediction.csv', index=False)


def get_final_table():
    current_working_directory = os.getcwd()
    loaded_df = pd.read_csv(f'{current_working_directory}/final_prediction.csv')
    return loaded_df



def get_nhl_players() -> pd.DataFrame:
    """Gets player IDs, teams, and positions from current NHL rosters."""
    base = "https://api-web.nhle.com/v1"
    response = requests.get(f"{base}/standings/now", timeout=30)
    response.raise_for_status()
    team_abbrevs = sorted({
        team["teamAbbrev"]["default"] for team in response.json()["standings"]
    })

    rows = []
    for team_abbrev in team_abbrevs:
        response = requests.get(f"{base}/roster/{team_abbrev}/current", timeout=30)
        response.raise_for_status()
        roster = response.json()
        for position_group in ("forwards", "defensemen", "goalies"):
            rows.extend(
                {
                    "player_id": player["id"],
                    "team_abbrev": team_abbrev,
                    "position": player["positionCode"],
                }
                for player in roster[position_group]
            )

    return pd.DataFrame(rows, columns=["player_id", "team_abbrev", "position"])
