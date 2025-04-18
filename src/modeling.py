import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import joblib

def import_data(path = "data/processed/Clean_AmesHousing.csv"):
    # import the data
    data = pd.read_csv(path)
    print(f"imported dataframe from {path}; {data.shape[0]} rows, {data.shape[1]} columns.")
    return data

def create_model(X_path= "data/train/X_train.csv", y_path= "data/train/y_train.csv"):
    # data = import_data(path)
    # making feature and target datasets
    X = import_data(X_path)
    y = import_data(y_path)
    y = y[y.columns.to_list()[0]]

    # train test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

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
    grid_search.fit(X, y)

    # get the best parameters
    print ("Best Parameteres:", grid_search.best_params_)
    print ("Best RMSE:", -grid_search.best_score_)

    # training models
    # # model_rf.fit(X_train, y_train)
    # # y_pred = model_rf.predict(X_test)

    model_rf_best = grid_search.best_estimator_
    model_rf_best.fit(X, y)
    y_pred = model_rf_best.predict(X)

    results= {
        "MAE": mean_absolute_error(y, y_pred),
        "MSE": mean_squared_error(y, y_pred),
        "R2 Score": r2_score(y, y_pred)
    }

    print(results)

    # save the model
    save_path = "models/randomForest_tunned.pkl"
    joblib.dump(model_rf_best, save_path)
    print(f"Saved the model at {save_path}")

def run_model(model_path= "models/randomForest_tunned.pkl", X_test_file = "data/test/X_test.csv", y_test_path = None, target_col = "SalePrice"):
    X = import_data(X_test_file)

    model = joblib.load(model_path)
    print(f"imported model from {model_path}")
    
    y_pred = model.predict(X)
    # print(type(y_pred))
    # print(len(y_pred))
    # print(y_pred)

    res = X.copy()
    if y_test_path != None:
        y = import_data(y_test_path)
        y = y[target_col]
        y = y.to_frame(name= "Actual_"+ target_col)
        res = pd.concat([res, y], axis=1)
        # print(f"concated actual {target_col} with features.")
        # print(res.head())
        results= {
            "MAE": mean_absolute_error(y, y_pred),
            "MSE": mean_squared_error(y, y_pred),
            "R2 Score": r2_score(y, y_pred)
        }
        print(f"result of actual values and predicted values of {target_col}.")
        print (results)
    else:
        print("No testing for prediction...")

    y_pred = pd.DataFrame(y_pred, columns= ["pred_"+target_col])
    # print("concating predicated values with dataset.")
    res = pd.concat([res, y_pred], axis= 1)
    # print(type(res))
    # print(res.head())
    return res

if __name__ == "__main__":
    print("---------- Running Modeling Script ----------")
    X_path= "data/train/X_train.csv"
    y_path= "data/train/y_train.csv" 
    X_test_file = "data/test/X_test.csv"
    y_test_path = "data/test/y_test.csv"
    model_path = "models/randomForest_tunned.pkl"
    
    create_model(X_path= X_path, y_path= y_path)
    run_model(model_path= model_path, X_test_file = X_test_file, y_test_path = y_test_path, target_col = "SalePrice")