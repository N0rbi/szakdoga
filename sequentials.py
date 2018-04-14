# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:25:02 2017

@author: Norbi
"""
from metrics import perplexity


def get_classifier(X_train, y_train, lstm_layers, units, embedding_units, number_of_chars):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout, Embedding
    classifier = Sequential()
    input_length = X_train.shape[1]
    classifier.add(Embedding(number_of_chars, embedding_units, input_length=input_length))
    for i in range(lstm_layers):
        classifier.add(LSTM(units=units, return_sequences=i != lstm_layers-1, recurrent_dropout=0.3))
        classifier.add(Dropout(rate=0.2))

    classifier.add(Dense(units=y_train.shape[1], activation='softmax'))
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy', perplexity])

    return classifier


def get_multitask_classifier(X_train, y_train, y_aux, number_of_chars,
                             embedding_units, units, lstm_layers):
    from keras.models import Model
    from keras.layers import Dense, LSTM, Dropout, Embedding, Input
    input_length = X_train.shape[1]
    input = Input(shape=(input_length,))
    x = Embedding(number_of_chars, embedding_units, input_length=input_length)(input)
    for i in range(lstm_layers):
        x = (LSTM(units=units, return_sequences=i != lstm_layers-1, recurrent_dropout=0.1))(x)
        x = Dropout(rate=0.2)(x)

    main_out = Dense(units=y_train.shape[1], activation='softmax')(x)
    aux_out = Dense(units=y_aux.shape[1], activation='softmax')(x)

    model = Model(inputs=[input], outputs=[main_out, aux_out])
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy', perplexity])

    return model
