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
import datetime
import fnmatch
import re
import keras

TARGET_DIR = 'target'
MODELS_DIR = os.path.join(TARGET_DIR, 'models')
LOG_DIR = os.path.join(TARGET_DIR, 'train_log')
ENCODERS_DIR = os.path.join(TARGET_DIR, 'encoders')


def read_file(file_content):
    with open(file_content, encoding='utf-8') as file_ref:
        content = file_ref.read()
    return content


class ModelArtifact:

    def __init__(self, artist=None, size_x=None, model_id=None):
        if not artist:
            return
        self.__id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "_" + artist \
            if not model_id else model_id
        # Create target dirs
        if not os.path.exists(TARGET_DIR):
            os.mkdir(TARGET_DIR)
            if not os.path.exists(MODELS_DIR):
                os.mkdir(MODELS_DIR)
            if not os.path.exists(LOG_DIR):
                os.mkdir(LOG_DIR)
            if not os.path.exists(ENCODERS_DIR):
                os.mkdir(ENCODERS_DIR)
        self.size_x = size_x

    def persist_model(self, model):
        serialized = model.to_json()
        if not os.path.exists(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        with open(os.path.join(MODELS_DIR, self.__id + ".json"), "w+") as json:
            json.write(serialized)

        model.save_weights(os.path.join(MODELS_DIR, self.__id+".h5"))

    def load_model(self, name):
        from keras.models import model_from_json
        loaded = open(os.path.join(MODELS_DIR, name+".json"), 'r')
        model = loaded.read()
        loaded.close()
        model = model_from_json(model)
        model.load_weights(os.path.join(MODELS_DIR, name+".h5"))

        return model

    def load_or_create_encoder(self, data, name=None):
        if not name:
            name = self.__id
        file_name = os.path.join(ENCODERS_DIR, '%s.pickle' % name)
        try:
            with open(file_name, 'rb') as file_stream:
                encoder = pickle.load(file_stream)
        except FileNotFoundError:
            encoder = CharEncoder()
            encoder.fit(data)
            if os.path.exists(file_name):
                raise FileExistsError
            with open(file_name, 'wb') as file_stream:
                pickle.dump(encoder, file_stream)
        return encoder

    def get_tensorflow_logdir(self):
        tf_dir = os.path.join(LOG_DIR, self.__id)
        if not os.path.exists(tf_dir):
            os.mkdir(tf_dir)
        return tf_dir

    def save_metadata_of_embedding(self, vocab):
        path = os.path.join(self.get_tensorflow_logdir(), 'metadata.tsv')
        with open(path, 'w') as file:
            buffer = "Word\tFrequency\n"
            for key, value in vocab.items():
                buffer += repr(key) + '\t' + str(value) + '\n'
            file.write(buffer)
        return path

    @staticmethod
    def show_all_artifacts():
        def list_files():
            ls = []
            for root, _, files in os.walk('.'):
                for filename in files:
                    ls.append(os.path.join(root, filename))
            return ls

        def extract_model_name(file_name):
            match = re.search('models/(.+?).h5$', file_name)
            return match.group(1) if match else None

        file_list = list_files()
        model_weights = fnmatch.filter(file_list, '*.h5')
        model_names = list(map(extract_model_name, model_weights))
        return model_names


class CharEncoder:

    def __init__(self):
        self.onehot = OneHotEncoder()
        self._RARE = '<RARE_CHAR>'
        self.vocab = dict()
        self._onehotted_vocab = dict()

    def transform(self, y, with_onehot=True):
        if not with_onehot:
            return np.array([self.vocab[ch] if ch in self.vocab else self.vocab[self._RARE].flatten() for ch in y])
        else:
            return np.array([self._onehotted_vocab[ch] if ch in self.vocab else self.vocab[self._RARE].flatten() for ch in y])

    def fit(self, y):
        vocab = sorted(list(set(y)))
        vocab.append(self._RARE)
        self.vocab = dict((c, i) for i, c in enumerate(vocab))

        self.onehot.fit(np.array(list(self.vocab.values())).reshape(-1, 1))
        for key, value in self.vocab.items():
            self._onehotted_vocab[key] = self.onehot.transform(value).toarray().flatten()

    def fit_transform(self, y, with_onehot=True):
        self.fit(y)
        return self.transform(y, with_onehot)

    def inverse_transform(self, y):
        rev_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join([rev_vocab[i] for i in y])


def save_metadata_of_embedding(path, vocab):
    with open(path, 'w+') as file:
        buffer = "Word\tFrequency\n"
        for key, value in vocab.items():
            buffer += strip_quotes(repr(key)) + '\t' + str(value) + '\n'
        file.write(buffer)


def strip_quotes(str):
    strip1 = str.strip("'")
    strip2 = str.strip('"')
    return strip1 if len(strip2) > len(strip1) else strip2


def read_batches(data, vocab_size, batch_size, seq_length, nb_epochs, encoder):
    length = data.shape[0]
    batch_chars = length // batch_size
    for _ in range(nb_epochs):
        for start in range(0, batch_chars - seq_length, seq_length):
            X = np.zeros((batch_size, seq_length))
            Y = np.zeros((batch_size, seq_length, vocab_size))
            for batch_idx in range(0, batch_size):
                for i in range(0, seq_length):
                    X[batch_idx, i] = data[batch_chars * batch_idx + start + i]
                    Y[batch_idx, i, data[batch_chars * batch_idx + start + i + 1]] = 1

            aux = list()
            for y in Y:
                aux.append([get_aux(yy, encoder) for yy in y])
            yield X, [Y, aux]


def get_aux(y, encoder):
    real_value = encoder.inverse_transform(y)
    result = []
    for char in real_value:
        result.append(int(is_vowel(char)))
    return np.array(result)


def is_vowel(y):
    return y in "aáeéiíoóöőuúüű"
