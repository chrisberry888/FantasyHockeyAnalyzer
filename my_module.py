import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

#Creates ML-model readable data from past year data
def merge_dataframes(arr, points_df):
    # remove "Fantasy_Points" column from dataframes in "arr"
    arr = [df.drop(columns=["Fantasy_Points"]) for df in arr]
    # merge all dataframes in "arr" on "Name" column
    df = arr[0]
    for i in range(1, len(arr)):
        df = pd.merge(df, arr[i], on='Name')
    # rename "Fantasy_Points" column to "Predicted_Fantasy_Points"
    points_df = points_df.rename(columns={"Fantasy_Points": "Predicted_Fantasy_Points"})
    # add "Predicted_Fantasy_Points" column from "points_df"
    df = pd.merge(df, points_df[['Name', 'Predicted_Fantasy_Points']], on='Name')
    return df

def separate_fantasy_points(df):
    fantasy_points = df['Predicted_Fantasy_Points'].tolist()
    df = df.drop(columns=['Predicted_Fantasy_Points'])
    return [df, fantasy_points]
    
#Gets rid of any string-based columns
def reformat_df(df):
    new_df = df.copy() # Make a copy of the input dataframe
    
    # Iterate over the columns of the dataframe
    for col in new_df.columns:
        # Check if the column name contains the substring "Name"
        if "Name" in col:
            # If it does, replace all the string values in that column with 0
            new_df[col] = new_df[col].apply(lambda x: 0 if isinstance(x, str) else x)
        else:
            # If it doesn't, replace all the string values in that column with integers
            new_df[col] = pd.Categorical(new_df[col]).codes
            
    return new_df
    
    
def get_name_predictions(regr, df):
    # Make a copy of the input dataframe
    new_df = df.copy()
    
    # Save the values in the "1_Name" column before reformatting
    names = new_df["Name"].values
    
    # Use the reformat_df function to reformat the dataframe
    new_df = reformat_df(new_df)
    
    # Use the regr object to make predictions on the reformatted dataframe
    predictions = regr.predict(new_df)
    
    # Create a new dataframe with the names and predictions
    result_df = pd.DataFrame({"Name": names, "Prediction": predictions})
    
    return result_df


def check_duplicate_names(df):
    names = df['Name']
    if len(names) != len(set(names)):
        duplicates = pd.Series(names).value_counts()
        duplicate_names = duplicates[duplicates > 1].index
        print("duplicate name: ",duplicate_names[0])
        return True
    else:
        return False



def sum_predictions(dfs):
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dfs)
    
    # Group by 'Name' and sum 'Prediction'
    summed_df = concatenated_df.groupby('Name')['Prediction'].sum().reset_index()
    
    return summed_df

def sim(regr, X, y, sims):
    models = []
    for i in range(sims):
        temp = clone(regr)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        temp.fit(X_train, y_train)
        models.append(temp)
    return models