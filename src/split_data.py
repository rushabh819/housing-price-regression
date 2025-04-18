import pandas as pd
from sklearn.model_selection import train_test_split

def import_data(path):
    return pd.read_csv(path)

def import_series(path):
    data = pd.read_csv(path)
    data = data[data.columns.tolist()[0]]
    return data

def split_dataset(path, target = "SalePrice"):
    data = import_data(path)
    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, random_state= 42)

    print(X_train.shape, type(X_train))
    print(X_test.shape, type(X_test))
    print(y_train.shape, type(y_train))
    print(y_test.shape, type(y_test))
    
    test_dest = "data/test"
    save_file(X_test, path = test_dest+"/X_test.csv")
    print(import_data(test_dest+"/X_test.csv").shape)
    save_file(y_test, path = test_dest+"/y_test.csv")
    print(import_series(test_dest+"/y_test.csv").shape , type(import_series(test_dest+"/y_test.csv")))

    train_dest = "data/train"
    save_file(X_train, path = train_dest+"/X_train.csv")
    print(import_data(train_dest+"/X_train.csv").shape)
    save_file(y_train, path = train_dest+"/y_train.csv")
    print(import_series(train_dest+"/y_train.csv").shape, type(import_series(train_dest+"/y_train.csv")))
    # print(import_series(train_dest+"/y_train.csv").head())

def save_file(data, path):
    data.to_csv(path, index = False)
    print(f"saved {path}")

if __name__ == "__main__":
    split_dataset(path = "data/processed/Clean_AmesHousing.csv")
