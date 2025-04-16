import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import joblib

# import the data
data = pd.read_csv("data/processed/Clean_AmesHousing.csv")

# making feature and target datasets
X = data.drop(columns='SalePrice')
y = data['SalePrice']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],            # Number of trees
    'max_depth': [10, 20, None],           # Depth of each tree
    'min_samples_split': [2, 5],           # Minimum samples to split a node
    'min_samples_leaf': [1, 2],            # Minimum samples at a leaf
    'max_features': ['sqrt', 'log2'],      # Number of features to consider at each split
    'bootstrap': [True]                    # Use bootstrapped samples
}


# Preparing Models dictionary
model_rf = RandomForestRegressor(random_state= 42)

# set up the grid
grid_search = GridSearchCV(
    estimator=model_rf,
    param_grid=param_grid,
    cv=5,                                  # 5-fold cross-validation
    scoring='neg_root_mean_squared_error', # RMSE (lower = better)
    verbose=1,
    n_jobs=-1                              # Use all available CPU cores'
)

# fit grid on training data
grid_search.fit(X_train, y_train)

# get the best parameters
print ("Best Parameteres:", grid_search.best_params_)
print ("Best RMSE:", -grid_search.best_score_)

# training models
# # model_rf.fit(X_train, y_train)
# # y_pred = model_rf.predict(X_test)

model_rf_best = grid_search.best_estimator_
model_rf_best.fit(X_train, y_train)
# y_pred = model_rf_best.predict(X_test)

# results= {
#     "MAE": mean_absolute_error(y_test, y_pred),
#     "MSE": mean_squared_error(y_test, y_pred),
#     "R2 Score": r2_score(y_test, y_pred)
# }

# print(results)

# save the model
save_path = "data/models/randomForest_tunned.pkl"
joblib.dump(model_rf_best, save_path)
print(f"Saved the model at {save_path}")