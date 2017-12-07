# -*- coding: utf-8 -*-
"""
Utility for deep learning.

Created on Sat Nov 25 17:15:11 2017

@author: Norbi
"""
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def read_file(file_content):
    with open(file_content, encoding='utf-8') as file_ref:
        content = file_ref.read()
    return content

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
