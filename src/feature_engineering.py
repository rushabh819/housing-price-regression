import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f'File now found on path: {file_path}')
        return None

def adding_features(dataFrame: pd.DataFrame):
    #adding engineered features to the dataset
    # 1. adding total square foot of house
    dataFrame['Total SF'] = dataFrame['1st Flr SF'] + dataFrame['2nd Flr SF'] + dataFrame['Total Bsmt SF']
    # 2. adding total number of bedrooms
    dataFrame['Total Bathrooms'] = dataFrame['Full Bath'] + 0.5 * dataFrame['Half Bath']
    # Bathroom per bedrooms
    dataFrame['bath per bedroom'] = dataFrame['Total Bathrooms'] / dataFrame['Bedroom AbvGr'].replace(0,1)
    # 4. House age
    dataFrame['House Age'] = dataFrame['Yr Sold'] - dataFrame['Year Built']

    print(f"Added new engineered features: total SF (total sq. feet), total bathrooms, bathroom per bedroom, house age")
    return dataFrame
def save_engineered_dataFrame(data: pd.DataFrame, save_path):
    data.to_csv(save_path, index=False)
    print(f'saved the dataset with new features at {save_path}')
    print(f'dimention of dataframe after feature engineering, {data.shape[0]} rows and {data.shape[1]}')

def run_feature_engineering(origin_path, dest_path):
    # load the data
    print("---------- Feature Engineernig ----------")
    data = load_data(file_path= origin_path) 
    new_data = adding_features(data)
    save_engineered_dataFrame(new_data, dest_path)
    print("---------- Feature Engineernig Done ----------")

if __name__ == "__main__":
    raw_path = "data/raw/AmesHousing.csv" 
    cleaned_path = "data/processed/Clean_AmesHousing.csv"
    run_feature_engineering(origin_path= cleaned_path, dest_path= cleaned_path)
