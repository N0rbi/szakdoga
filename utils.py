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
import fnmatch, re

TARGET_DIR = 'target'
MODELS_DIR = os.path.join(TARGET_DIR, 'models')
LOG_DIR = os.path.join(TARGET_DIR, 'train_log')
ENCODERS_DIR = os.path.join(TARGET_DIR, 'encoders')


def read_file(file_content):
    with open(file_content, encoding='utf-8') as file_ref:
        content = file_ref.read()
    return content


class ModelArtifact:

    def __init__(self, artist, model_id=None):
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

    def persist_model(self, model):
        serialized = model.to_json()
        if not os.path.exists(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        with open(os.path.join(MODELS_DIR, self.__id + ".json"), "w+") as json:
            json.write(serialized)

        model.save_weights(os.path.join(MODELS_DIR, self.__id+".h5"))

    def load_model(self):
        from keras.models import model_from_json
        loaded = open(os.path.join(MODELS_DIR, self.__id+".json"), 'r')
        model = loaded.read()
        loaded.close()
        model = model_from_json(model)
        model.load_weights(os.path.join(MODELS_DIR, self.__id+".h5"))

        return model

    def load_or_create_encoder(self, data):
        file_name = os.path.join(ENCODERS_DIR, '%s.pickle' % self.__id)
        try:
            with open(file_name, 'rb') as file_stream:
                encoder = pickle.load(file_stream)
        except FileNotFoundError:
            encoder = CharEncoder(ENCODER_FORMAT_LOWERCASE)
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


def save_embedding_to_board(tensorboard_callback, batch_no):
    tensorboard_callback.saver.save(tensorboard_callback.sess,
                                    tensorboard_callback.embeddings_ckpt_path,
                                    batch_no)


class DataChunk:

    def __init__(self, data, steps, chunk_size, encoder):
        self.__data = data
        self.__steps = steps
        self.__chunk_size = chunk_size
        self.__chunks = range(0, len(data)-self.__steps, self.__chunk_size)
        self.__encoder = encoder

    def __iter__(self):
        '''
        A generator that returns the current batch of data.
        '''
        for u in self.__chunks:
            train_X = []
            train_y = []
            for i in range(u, u+self.__chunk_size-self.__steps+1):
                if i + self.__steps == len(self.__data):
                    return
                current_in = self.__data[i:i + self.__steps]
                current_out = self.__data[i + self.__steps]
                train_X.append(self.__encoder.transform(current_in, False))
                train_y.append(self.__encoder.transform(current_out).flatten())

            train_X = np.array(train_X)
            train_y = np.array(train_y)

            yield train_X, train_y
        return


ENCODER_FORMAT_LOWERCASE = {'lowercase': lambda t: t.lower()}


class CharEncoder:

    def __init__(self, formaters=dict()):
        self.onehot = OneHotEncoder()
        self._RARE = '<RARE_CHAR>'
        self.vocab = dict()
        self._onehotted_vocab = dict()
        self._formaters = formaters

    def _format(self, y):
        for _, formater in self._formaters.items():
            y = formater(y)
        return y

    def transform(self, y, with_onehot=True):
        y = self._format(y)
        if not with_onehot:
            return np.array([self.vocab[ch] if ch in self.vocab else self.vocab[self._RARE].flatten() for ch in y])
        else:
            return np.array([self._onehotted_vocab[ch] if ch in self.vocab else self.vocab[self._RARE].flatten() for ch in y])

    def fit(self, y):
        y = self._format(y)
        vocab = sorted(list(set(y)))
        vocab.append(self._RARE)
        self.vocab = dict((c, i) for i, c in enumerate(vocab))

        self.onehot.fit(np.array(list(self.vocab.values())).reshape(-1, 1))
        for key, value in self.vocab.items():
            self._onehotted_vocab[key] = self.onehot.transform(value).toarray().flatten()

    def fit_transform(self, y, with_onehot=True):
        self.fit(y)
        y = self._format(y)
        return self.transform(y, with_onehot)

    def inverse_transform(self, y):
        rev_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join([rev_vocab[i] for i in y])

    def get_formaters_str(self):
        return get_formaters_str(self._formaters)


def get_formaters_str(format_dict):
    """
    with the same formaters you should get the same string
    :return: formater identifier
    """
    return ''.join(sorted(list(map(lambda key: key[0], [key for key, _ in format_dict.items()]))))
