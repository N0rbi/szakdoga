# -*- coding: utf-8 -*-
"""
Utility for deep learning.

Created on Sat Nov 25 17:15:11 2017

@author: Norbi
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
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

class DataChunk:
    
    def __init__(self, data, steps, chunk_size):
        self.__data = data
        self.__steps = steps
        self.__chunk_size = chunk_size
        self.__has_no_remainer = len(self.__data) % self.__chunk_size == 0
        self.__chunks = range(self.__steps, len(data), self.__chunk_size)
    
    def __iter__(self):
        '''
        A generator that returns the current batch of files.
        '''
        for u in self.__chunks if self.__has_no_remainer else self.__chunks[:-1]:
            train_X = []
            train_y = []
            for i in range(u-self.__steps, u):
                current_in = self.__data[i:i+ self.__steps]
                current_out = self.__data[i + self.__steps]
                train_X.append(current_in)
                train_y.append(current_out)
            
            train_X = np.array(train_X)
            train_y = np.array(train_y)
        
            yield train_X, train_y
        return
    
    def get_dummy(self):
        '''
        A function that returns a structure of vectors of zero. The structure
        is the same shape as the generator's return value's.
        It is useful for the neural network to fit the input size.
        '''
        dummy_var = np.zeros(self.__data[0].shape)
        dummy_X = []
        dummy_y = []
        for i in range(self.__chunk_size):
            current_in = np.repeat(np.array([dummy_var]), [self.__steps], axis=0)
            current_out = dummy_var
            dummy_X.append(current_in)
            dummy_y.append(current_out)
        dummy_X = np.array(dummy_X)
        dummy_y = np.array(dummy_y)
        
        return dummy_X, dummy_y

class CharEncoder:
    
    def __init__(self):
        self.onehot = OneHotEncoder()
        self._RARE = '<RARE_CHAR>'
        
    def transform(self, y):
        return np.array([self._vocab[ch] if ch in self._vocab else self._vocab[self._RARE].flatten() for ch in y])
    
    def fit(self, y):
        vocab = sorted(list(set(y)))
        vocab.append(self._RARE)
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
