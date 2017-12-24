# -*- coding: utf-8 -*-

import numpy as np
from utils import CharEncoder, read_file, persist_model, DataChunk
from sequentials import get_classifier_1lstm_short_mem
from keras.callbacks import ModelCheckpoint, TensorBoard
import os

data = read_file('dataset/30y.txt')
encoder = CharEncoder()
data = encoder.fit_transform(data)
chunk = DataChunk(data, 100, 100)
classifier = get_classifier_1lstm_short_mem(*chunk.get_dummy())

# define the checkpoint
filepath=os.path.join("checkpoints","weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=250, write_graph=True, write_images=False)
callbacks_list = [checkpoint, tensorboard]
for i, (train_X, train_y) in enumerate(iter(chunk)):
    print('%d. chunk is being trained on.'% i)
    classifier.fit(train_X, train_y, epochs=10, batch_size=250, validation_split=0.2, callbacks=callbacks_list)


# Validating on real life data

sample = '''A buli íze a számban
Lányok neonruhában
Oldódnak a színpadon
Részeg vagyok, nem is tudom
Mire jó, ha jó ez'''

sample = encoder.transform(sample)

def predict_next(text):
    x = text[-100:]
    x = np.array([x])
    probabilities = classifier.predict(x)
    y = np.array([np.random.choice(len(prob), p=prob) for prob in probabilities])
    y = encoder.onehot.transform(np.array(y).reshape(-1,1)).toarray()
    text = np.append(text, y, axis=0)
    return text

def predict_next_n(text, n):
    if n == 0:
        return text
    else:
        return predict_next_n(predict_next(text), n-1)


prediction = predict_next_n(sample, 400)

result = []

for yc in prediction:
    yc = yc.reshape(1,-1)
    yc = encoder.inverse_transform(yc)
    result.append(yc)
    
print(''.join(result))