This file provides individual summaries of each core module used in the regression pipeline.

---

## `data_wrangling.py`
- Loads raw housing data
- Drops columns with high missing value ratios
- Fills missing values using median (numeric) and mode (categorical)
- Saves cleaned dataset to `data/processed`

---

## `feature_engineering.py`
- Adds new domain-relevant features:
  - `Total SF`, `Total Bathrooms`, `Bath per Bedroom`, `House Age`
- Reads from and saves back to the processed dataset

---

## `split_data.py`
- Splits cleaned dataset into train and test sets (80/20 split)
- Saves `X_train`, `X_test`, `y_train`, `y_test` into respective folders

---

## `scaling_features.py`
- One-hot encodes categorical features
- Converts boolean columns to integers
- Handles infinite values and applies `StandardScaler` to numeric features
- Combines all features with the target and saves the final scaled dataset

---

## `modeling.py`
- Trains and evaluates multiple regression models (RF, Linear, Ridge, Lasso)
- Performs hyperparameter tuning using `GridSearchCV`
- Prints best parameters and metrics (MAE, MSE, R2)
- Saves trained model to `models/randomForest_tunned.pkl`
- Optionally predicts and evaluates on test set