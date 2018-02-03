# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:25:02 2017

@author: Norbi
"""

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

def get_classifier_1lstm_short_mem(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    classifier = Sequential()
    classifier.add(LSTM(units = 10, input_shape=X_train.shape[1:3]))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(Dense(units = y_train.shape[1], activation='softmax'))
    
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])
    
    return classifier

def get_classifier_2lstm_medium_mem(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    classifier = Sequential()
    classifier.add(LSTM(units = 125, input_shape=X_train.shape[1:3], return_sequences=True))
    classifier.add(Dropout(rate=0.2))
    classifier.add(LSTM(units = 125))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units = y_train.shape[1], activation='softmax'))
    
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    
    return classifier
