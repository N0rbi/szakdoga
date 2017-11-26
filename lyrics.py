# -*- coding: utf-8 -*-

import numpy as np
from utils import CharEncoder, read_file

def get_classifier_4lstm_4do(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    classifier = Sequential()
    classifier.add(LSTM(units = 50, return_sequences=True, input_shape=X_train.shape[1:3]))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(LSTM(units = 50, return_sequences=True))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(LSTM(units = 50, return_sequences=True))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(LSTM(units = 50, return_sequences=False))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(Dense(units = y_train.shape[1], activation='softmax'))
    
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy")

    return classifier

def get_classifier_1lstm_long_mem(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    classifier = Sequential()
    classifier.add(LSTM(units = 256, input_shape=X_train.shape[1:3]))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(Dense(units = y_train.shape[1], activation='softmax'))
    
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy")

    return classifier

data = read_file('dataset/30y.txt')

encoder = CharEncoder()

data = encoder.fit_transform(data)

train_X = []
train_y = []
for i in range(0, len(data) - 100):
    current_in = data[i:i+100]
    current_out = data[i + 100]
    train_X.append(current_in)
    train_y.append(current_out)

train_X = np.array(train_X)
train_y = np.array(train_y)


classifier = get_classifier_1lstm_long_mem(train_X, train_y)

classifier.fit(train_X, train_y, epochs=60, batch_size=1000)

serialized = classifier.to_json()

with open("model_2.json", "w") as json:
    json.write(serialized)
    
classifier.save_weights("model_2.h5")