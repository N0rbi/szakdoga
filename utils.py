# -*- coding: utf-8 -*-
"""
Utility for deep learning.

Created on Sat Nov 25 17:15:11 2017

@author: Norbi
"""
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

def read_file(file_content):
    with open(file_content, encoding='utf-8') as file_ref:
        content = file_ref.read()
    return content

def persist_model(model, name):
    serialized = model.to_json()

    with open(os.path.join("models", name+".json"), "w") as json:
        json.write(serialized)
    
    model.save_weights(os.path.join("models", name+".h5"))
    
def load_model(name):
    from keras.models import model_from_json
    loaded = open(os.path.join("models", name+".json"), 'r')
    model = loaded.read()
    loaded.close()
    model = model_from_json(model)
    model.load_weights(os.path.join("models", name+".h5"))
    
    return model


def get_chunk(data, steps, chunk_size):
    unit = int(len(data) / chunk_size)
    has_no_remainer = len(data) % chunk_size == 0
    chunks = range(unit, len(data), unit)
    for u in chunks if has_no_remainer else chunks[:-1]:
        train_X = []
        train_y = []
        for i in range(u-unit, u):
            current_in = data[i:i+steps]
            current_out = data[i + steps]
            train_X.append(current_in)
            train_y.append(current_out)
        
        train_X = np.array(train_X)
        train_y = np.array(train_y)
    
        yield train_X, train_y

class CharEncoder:
    
    def __init__(self):
        self.onehot = OneHotEncoder()
        
    def transform(self, y):
        return np.array([self._vocab[ch].flatten() for ch in y])
    
    def fit(self, y):
        vocab = sorted(list(set(y)))
        self._vocab = dict((c,i) for i, c in enumerate(vocab))
        self.onehot.fit(np.array(list(self._vocab.values())).reshape(-1, 1))
        for key, value in self._vocab.items():
            self._vocab[key] = self.onehot.transform(value).toarray().flatten()
        
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        rev_vocab = {self._invert_onehot(v): k for k, v in self._vocab.items()}
        return ''.join([rev_vocab[self._invert_onehot(i)] for i in y])

    def _invert_onehot(self, encoded):
        return np.where(encoded==1)[0][0]
