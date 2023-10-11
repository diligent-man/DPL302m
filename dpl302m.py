# -*- coding: utf-8 -*-
"""DPL302m

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QKvF7xOqvEILUKatDJVDL5PE1BwM_tRV
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#load data
data = pd.read_csv('Housing.csv')
data.head(5)

train_data, dev_data = train_test_split(data[:300], test_size=0.1, random_state=42)
dev_data, test_data = train_test_split(dev_data, test_size=0.5, random_state=42)

features = ['hotwaterheating', 'airconditioning','parking', 'prefarea', 'furnishingstatus']
target = 'price'

X_train = train_data[features]
y_train = train_data[target]
X_dev = dev_data[features]
y_dev = dev_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Build DNN model
def house_pricing():
    model = Sequential()
    model.add(Dense(64, input_dim=len(features), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
return model
comparator(summary(house_pricing), output)
house_pricing.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
