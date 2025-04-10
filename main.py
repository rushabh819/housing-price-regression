import pandas as pd
import numpy as np

import src.feature_engineering as fe
import src.data_wrangling as dw
import src.scaling_features as sf

def main():
    raw_path = "data/raw/AmesHousing.csv"
    cleaned_path = "data/processed/Clean_AmesHousing.csv"
    # Data Wrangling
    dw.run_data_wrangling(raw_path= raw_path, cleaned_path= cleaned_path)
    # Adding new Features with Feature Engineering
    fe.run_feature_engineering(origin_path= cleaned_path, dest_path= cleaned_path)
    # Scaling and encoding while converting bool into integer    
    sf.runScaling_features(origin_path= cleaned_path, dest_path= cleaned_path)

    df_final_cleaned = pd.read_csv(cleaned_path)

main()