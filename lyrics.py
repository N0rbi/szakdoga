# -*- coding: utf-8 -*-

import numpy as np
from utils import CharEncoder, read_file, persist_model, get_chunk
from sequentials import get_classifier_1lstm_short_mem
from keras.callbacks import ModelCheckpoint, TensorBoard
import os

data = read_file('dataset/30y.txt')
encoder = CharEncoder()
data = encoder.fit_transform(data)
chunks = enumerate(get_chunk(data, 100, 100))

_, (train_X, train_y) = next(chunks)
classifier = get_classifier_1lstm_short_mem(train_X, train_y)


# define the checkpoint
filepath=os.path.join("checkpoints","weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=250, write_graph=True, write_images=False)
callbacks_list = [checkpoint, tensorboard]

# reseting the generator
chunks = enumerate(get_chunk(data, 100, 100))

for i, (train_X, train_y) in chunks:
    print('%d. chunk is being trained on.'% i)
    classifier.fit(train_X, train_y, epochs=10, batch_size=250, validation_split=0.2, callbacks=callbacks_list)


sample = '''A buli íze a számban
Lányok neonruhában
Oldódnak a színpadon
Részeg vagyok, nem is tudom
Mire jó, ha jó ez
Ha az alkohol boldoggá tesz
Akkor az a kevés kis öröm
Is kihányva fekszik a kövön

Param, pararam
Pararara-rararam
Param, pararam
Á-há

(Refrén 2×:)
Fáj a fejem, a szívem túl nagy
És nem tudom, nem tudom, hol vagy
Forog a világ, elfolyik minden
Nekem senkim, de senkim sincse'''

sample = encoder.transform(sample)

out = []

for i in range(100, len(sample)):
    s = sample[i-100:i]
    out.append(s)

out = np.array(out)

y = classifier.predict(out)
#y = np.argmax(y, axis=1)
y = np.array([np.random.choice(len(single_prediction), p=single_prediction) for single_prediction in y])

result = []

for yc in y:
    yc = encoder.onehot.transform(yc).toarray()
    yc = encoder.inverse_transform(yc)
    result.append(yc)
    
print(''.join(result))