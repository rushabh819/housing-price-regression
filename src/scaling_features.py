import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f'File now found on path: {file_path}')
        return None

def one_hot_encode_cat(X_dataFrame:pd.DataFrame):
    encode_cat = pd.get_dummies(X_dataFrame)
    print("One Hot Encoding done for Cat columns.")
    return encode_cat

def handling_inf_values(data: pd.DataFrame):
    # take the numeric data from dataframer
    num_data = data.select_dtypes(include=['int64', 'float64']).copy()
    num_data = num_data.replace([np.inf, -np.inf], np.nan)
    return pd.concat([num_data, data.select_dtypes(exclude=['int64', 'float64'])], axis= 1)

def scaling_num(X_dataFrame:pd.DataFrame):
    scaler = StandardScaler()
    X_dataFrame_num = X_dataFrame.select_dtypes(include=['int64', 'float64'])
    X_dataFrame_ex_num = X_dataFrame.select_dtypes(exclude=['int64', 'float64'])
    X_dataFrame_num = pd.DataFrame(scaler.fit_transform(X_dataFrame_num), columns= X_dataFrame_num.columns)
    df = pd.concat([X_dataFrame_num, X_dataFrame_ex_num], axis= 1)
    print("Scaling for Num Columns completed.")
    return df

def convert_bool_to_int(data: pd.DataFrame):
    data_only_bool = data.select_dtypes(include= ["bool"])
    data_only_bool = data_only_bool.astype("int64")
    print("Converted boolean columns into int columns")
    df = pd.concat([data.select_dtypes(exclude=['bool']), data_only_bool], axis= 1)
    return df

def save_data(dataFrame:pd.DataFrame, dest_file_path):
    dataFrame.to_csv(dest_file_path, index=False)
    print(f"Refined Dataframe saved to {dest_file_path}")

def runScaling_features(origin_path, dest_path):
    print("--------------- Scaling features started ---------------")
    # load dataset
    dataFrame = load_data(file_path= origin_path)

    # making X and y from the dataset; target column is "SalePrice"
    X = dataFrame.drop(columns="SalePrice")
    y = dataFrame[["SalePrice"]]

    # scaling categorical columns
    X = one_hot_encode_cat(X)

    # handling inf values
    X = handling_inf_values(X)

    # scaling number columns
    X = scaling_num(X)

    # convert bool into int
    X = convert_bool_to_int(X)

    # df = X.merge(y)
    df = pd.concat([pd.DataFrame(X) , y], axis= 1)

    # print(df.isna().any().any())

    save_data(df, dest_file_path= dest_path)

    print("--------------- Scaling features Finished ---------------")

if __name__ == '__main__':
    from_path = "data/processed/Clean_AmesHousing.csv"
    to_path = "data/processed/Cleaned_Scaled_AmesHousing.csv"

    runScaling_features(origin_path= from_path, dest_path=to_path)