import src.data_wrangling as dw 
import src.feature_engineering as fe
import src.scaling_features as sf
import pandas as pd

def main():
    raw_path = "data/raw/AmesHousing.csv"
    cleaned_path = "data/processed/Clean_AmesHousing.csv"

    fe.run_feature_engineering(origin_path= raw_path, dest_path= cleaned_path)    
    dw.run_data_wrangling(raw_path= cleaned_path, cleaned_path= cleaned_path)
    sf.runScaling_features(origin_path= cleaned_path, dest_path= cleaned_path)

    data = pd.read_csv(cleaned_path)
    print(data.head())
main()