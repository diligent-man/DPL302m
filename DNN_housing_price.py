import pandas as pd
import numpy as np
import tensorflow as tf


def feature_engineering(df):
    non_categorial_features_1 = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
    non_categorial_features_2 = ['furnishingstatus']
    return df


def feature_selection(df):
    print(df.corr()["price"].sort_values())
    return 



def main() -> None:
    df = pd.read_csv('Housing.csv')
    df = feature_engineering(df)
    features = feature_selection(df)

    
    


    # print(df.corr())
    return None


if __name__ == '__main__':
    main()