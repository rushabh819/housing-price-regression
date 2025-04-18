import pandas as pd
import numpy as np

import src.feature_engineering as fe
import src.data_wrangling as dw
import src.scaling_features as sf
import src.modeling as model

def main():
    raw_path = "data/raw/AmesHousing.csv"
    cleaned_path = "data/processed/Clean_AmesHousing.csv"
    
    # Data Wrangling
    dw.run_data_wrangling(raw_path= raw_path, cleaned_path= cleaned_path)
    # Adding new Features with Feature Engineering
    fe.run_feature_engineering(origin_path= cleaned_path, dest_path= cleaned_path)
    # Scaling and encoding while converting bool into integer    
    sf.runScaling_features(origin_path= cleaned_path, dest_path= cleaned_path)
    # running the model
    X_test_file = "data/test/X_test.csv"
    y_test_path = "data/test/y_test.csv"
    model_path = "models/randomForest_tunned.pkl"

    res = model.run_model(model_path= model_path, X_test_file = X_test_file, y_test_path = y_test_path, target_col = "SalePrice")
    print("--------------- RESULT --------------")
    print(res.head())

main()