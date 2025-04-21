import pandas as pd
import numpy as np

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f'File now found on path: {file_path}')
        return None

def remove_null_columns(data: pd.DataFrame, threshold = 0.9):
    null_mean = data.isnull().mean()
    drop_null_index = null_mean[null_mean > threshold].index
    data.drop(columns = drop_null_index, inplace = True)
    print(f"Dropped {len(drop_null_index)} columns with missing values > {threshold*100}%")
    return data

def filling_missing_values(data: pd.DataFrame):
    for col in data.columns:
        # check if the column has null records or not
        if data[col].isnull().sum() > 0:
            # check is it is catogerical or numerical:
            if data[col].dtype == 'object':
                # if it is categorical, fill with mode
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                # if it is numerical, fill with median
                data[col] = data[col].fillna(data[col].median())
    print("Missing values are filled (media for numerical and mode for categorical)")
    return data


def save_data(data: pd.DataFrame, file_path):
    data.to_csv(file_path, index = False)
    print(f"cleaned data saved to: {file_path}")
    print(f"Data saved: {data.shape[0]} rows, {data.shape[1]} columns")

def run_data_wrangling(raw_path= "data/raw/AmesHousing.csv", cleaned_path= "data/processed/Clean_AmesHousing.csv"):
    print("------------ Data Wrangling Started ------------")
    df = load_data(raw_path)
    if df is not None:
        df = remove_null_columns(df)
        df = filling_missing_values(df)
        save_data(df, cleaned_path)
    print("------------- Data Wrangling Finished ------------")

if __name__ == "__main__":
    raw_path = "data/raw/AmesHousing.csv"
    cleaned_path = "data/processed/Clean_AmesHousing.csv"
    run_data_wrangling(raw_path= raw_path, cleaned_path= cleaned_path)