import src.data_wrangling as dw 
import src.feature_engineering as fe
import pandas as pd

def main():
    raw_path = "data/raw/AmesHousing.csv"
    cleaned_path = "data/processed/Clean_AmesHousing.csv"
    
    dw.run_data_wrangling(raw_path= raw_path, cleaned_path= cleaned_path)
    fe.run_feature_engineering(cleaned_path= cleaned_path)

    # data = pd.read_csv(raw_path)
    # print(data.head())
main()