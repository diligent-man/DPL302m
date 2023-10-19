import pandas as pd
import numpy as np
import tensorflow as tf


def feature_engineering(df):
    non_categorial_features_1 = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
    non_categorial_features_2 = ['furnishingstatus']

    # encoding
    df[non_categorial_features_1] = df[non_categorial_features_1].replace(['yes', 'no'], [0, 1])
    df[non_categorial_features_2] = df[non_categorial_features_2].replace(['furnished', 'semi-furnished', 'unfurnished'], [0, 1, 1])
    return df


def feature_selection(df):
    return df.corr()["price"].sort_values(ascending=False).iloc[:6]


def model_init(inp_shape):
    inputs = keras.Input(shape=inp_shape, name='inp_layer')
    layer_1 = keras.layers.Dense(512, activation='relu', name='first_layer')(inputs)
    layer_2 = keras.layers.Dense(216, activation='relu', name='second_layer')(layer_1)
    outputs = keras.layers.Dense(10, activation='softmax', name='output_layer')(layer_2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model




def main() -> None:
    df = pd.read_csv('Housing.csv')
    df = feature_engineering(df)
    features = feature_selection(df).index.to_list()
    df = df[features]
    df['price'] = df['price'].apply(lambda x: (x - df['price'].mean) / df['price'].std)

    return None


if __name__ == '__main__':
    main()