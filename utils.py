# -*- coding: utf-8 -*-
"""
Utility for deep learning.

Created on Sat Nov 25 17:15:11 2017

@author: Norbi
"""
from sklearn.preprocessing import OneHotEncoder
#for pickling lambdas
import dill as pickle
import numpy as np
import os
import tensorflow as tf

MODELS_DIR = 'models'
LOG_DIR = 'train_log'
ENCODERS_DIR = 'encoders'

def read_file(file_content):
    with open(file_content, encoding='utf-8') as file_ref:
        content = file_ref.read()
    return content


def persist_model(model, name):
    serialized = model.to_json()
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    with open(os.path.join(MODELS_DIR, name+".json"), "w") as json:
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


def persist_object(obj, name):
    file_name = '%s.pickle' % name
    if os.path.exists(file_name):
        raise FileExistsError
    with open(file_name, 'wb') as file_stream:
        pickle.dump(obj, file_stream)


def load_object(name):
    with open('%s.pickle' % name, 'rb') as file_stream:
        return pickle.load(file_stream)


def write_log_to_board(tensorboard_callback, names, logs, batch_no):
    """
    [Source](https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11#file-demo-py-L24)
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        tensorboard_callback.writer.add_summary(summary, batch_no)
        tensorboard_callback.writer.flush()


class DataChunk:
    
    def __init__(self, data, steps, chunk_size):
        self.__data = data
        self.__steps = steps
        self.__chunk_size = chunk_size
        self.__chunks = range(0, len(data)-self.__steps, self.__chunk_size)
        print('')
    
    def __iter__(self):
        '''
        A generator that returns the current batch of files.
        '''
        for u in self.__chunks:
            train_X = []
            train_y = []
            for i in range(u, u+self.__chunk_size-self.__steps+1):
                if i + self.__steps == len(self.__data):
                    return
                current_in = self.__data[i:i + self.__steps]
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


ENCODER_FORMAT_LOWERCASE = {'lowercase': lambda t: t.lower()}


class CharEncoder:
    
    def __init__(self, formaters=dict()):
        self.onehot = OneHotEncoder()
        self._RARE = '<RARE_CHAR>'
        self._vocab = dict()
        self._formaters = formaters

    def _format(self, y):
        for _, formater in self._formaters.items():
            y = formater(y)
        return y

    def transform(self, y):
        y = self._format(y)
        return np.array([self._vocab[ch] if ch in self._vocab else self._vocab[self._RARE].flatten() for ch in y])
    
    def fit(self, y):
        y = self._format(y)
        vocab = sorted(list(set(y)))
        vocab.append(self._RARE)
        self._vocab = dict((c,i) for i, c in enumerate(vocab))
        
        self.onehot.fit(np.array(list(self._vocab.values())).reshape(-1, 1))
        for key, value in self._vocab.items():
            self._vocab[key] = self.onehot.transform(value).toarray().flatten()
    
    def fit_transform(self, y):
        self.fit(y)
        y = self._format(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        rev_vocab = {self._invert_onehot(v): k for k, v in self._vocab.items()}
        return ''.join([rev_vocab[self._invert_onehot(i)] for i in y])

    @staticmethod
    def _invert_onehot(encoded):
        return np.where(encoded == 1)[0][0]

    def get_formaters_str(self):
        return get_formaters_str(self._formaters)


def get_formaters_str(format_dict):
    """
    with the same formaters you should get the same string
    :return: formater identifier
    """
    return ''.join(sorted(list(map(lambda key: key[0], [key for key, _ in format_dict.items()]))))


def load_or_create_encoder(artist, formatter, data):
    filename = '%s/%s_encoder_%s' % (ENCODERS_DIR, artist, get_formaters_str(formatter))
    try:
        encoder = load_object(filename)
    except FileNotFoundError:
        encoder = CharEncoder(ENCODER_FORMAT_LOWERCASE)
        encoder.fit(data)
        if not os.path.exists(ENCODERS_DIR):
            os.mkdir(ENCODERS_DIR)
        persist_object(encoder, filename)
    return encoder


class TrainLogger(object):
    '''
    Training logger class was pulled from ekzhang's repo on char rnn for keras
    https://github.com/ekzhang/char-rnn-keras/blob/master/train.py
    '''
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc,v loss, v acc\n')

    def add_entry(self, loss, acc, v_loss, v_acc):
        self.epochs += 1
        s = '{},{},{},{},{}\n'.format(self.epochs, loss, acc, v_loss, v_acc)
        with open(self.file, 'a') as f:
            f.write(s)