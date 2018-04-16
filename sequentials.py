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
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy', perplexity])

    return classifier


def get_multitask_classifier(batch_size, seq_len, vocab_size, aux_vocab_size, layers, embedding, units):
    from keras.models import Model
    from keras.layers import Dense, LSTM, Dropout, Embedding, Input, TimeDistributed, Activation
    input = Input(batch_shape=(batch_size, seq_len))
    x = Embedding(vocab_size, embedding)(input)
    for i in range(layers):
        x = (LSTM(units, return_sequences=True, stateful=True))(x)
        x = Dropout(rate=0.2)(x)

    main_out = TimeDistributed(Dense(vocab_size))(x)
    aux_out = TimeDistributed(Dense(aux_vocab_size))(x)
    main_out = Activation('softmax', name='main_out')(main_out)
    aux_out = Activation('softmax', name='aux_out')(aux_out)
    model = Model(inputs=[input], outputs=[main_out, aux_out])
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy', perplexity])

    return model
