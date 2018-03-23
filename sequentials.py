# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:25:02 2017

@author: Norbi
"""
from metrics import perplexity


def get_classifier(X_train, y_train, lstm_layers, units, embedding_units):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout, Embedding
    classifier = Sequential()
    batch_size = X_train.shape[0]
    input_length = X_train.shape[1]
    classifier.add(Embedding(batch_size, embedding_units, input_length=input_length))
    for i in range(lstm_layers):
        classifier.add(LSTM(units=units, return_sequences=i != lstm_layers-1, recurrent_dropout=0.3))
        classifier.add(Dropout(rate=0.2))

    classifier.add(Dense(units=y_train.shape[1], activation='softmax'))
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy', perplexity])

    return classifier
