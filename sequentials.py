# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:25:02 2017

@author: Norbi
"""
from metrics import perplexity


def get_classifier(batch_size, seq_len, vocab_size, layers, embedding, units):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout, Embedding, TimeDistributed, Activation
    classifier = Sequential()
    classifier.add(Embedding(vocab_size, embedding, batch_input_shape=(batch_size, seq_len)))
    for i in range(layers):
        classifier.add(LSTM(units, return_sequences=True, stateful=True))
        classifier.add(Dropout(0.2))

    classifier.add(TimeDistributed(Dense(vocab_size)))
    classifier.add(Activation('softmax'))
    classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy', perplexity])

    return classifier
