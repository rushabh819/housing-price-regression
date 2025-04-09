import pandas as pd
import data_wrangling

def adding_features(dataFrame):
    #adding engineered features to the dataset
    # 1. adding total square foot of house
    dataFrame['Total SF'] = dataFrame['1st Flr SF'] + dataFrame['2nd Flr SF'] + dataFrame['Total Bsmt SF']
    # 2. adding total number of bedrooms
    dataFrame['Total Bathrooms'] = dataFrame['Full Bath'] + 0.5 * dataFrame['Half Bath']
    # Bathroom per bedrooms
    dataFrame['bath per bedroom'] = dataFrame['Total Bathrooms'] / dataFrame['Bedroom AbvGr']
    # 4. House age
    dataFrame['House Age'] = dataFrame['Yr Sold'] - dataFrame['Year Built']

    print(f"Added new engineered features: total SF (total sq. feet), total bathrooms, bathroom per bedroom, house age")
    return dataFrame
def save_engineered_dataFrame(data, save_path):
    data.to_csv(save_path, index=False)
    print(f'saved the dataset with new features at {save_path}')
    print(f'dimention of dataframe after feature engineering, {data.shape[0]} rows and {data.shape[1]}')

def run_feature_engineering(cleaned_path):
    # load the data
    data = data_wrangling.load_data(cleaned_path) # taking load data method from data_wrangling module
    new_data = adding_features(data)
    # save the data
    save_engineered_dataFrame(new_data, cleaned_path)

if __name__ == "__main__":
    raw_path = "data/raw/AmesHousing.csv" 
    cleaned_path = "data/processed/Clean_AmesHousing.csv"
    data_wrangling.run_data_wrangling(raw_path, cleaned_path)
    run_feature_engineering(raw_path= raw_path, cleaned_path=cleaned_path)
