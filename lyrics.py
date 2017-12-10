# -*- coding: utf-8 -*-

import numpy as np
from utils import CharEncoder, read_file, persist_model, get_chunk
from sequentials import get_classifier_1lstm_short_mem

data = read_file('dataset/30y.txt')
encoder = CharEncoder()
data = encoder.fit_transform(data)
chunks = enumerate(get_chunk(data, 100, 100))

_, (train_X, train_y) = next(chunks)
classifier = get_classifier_1lstm_short_mem(train_X, train_y)

for i, (train_X, train_y) in chunks:
    print('%d. chunk is being trained on.'% i)
    classifier.fit(train_X, train_y, epochs=45, batch_size=250, validation_split=0.2)


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

y = np.argmax(y, axis=1) #NOT AXIS 0

result = []

for yc in y:
    yc = encoder.onehot.transform(yc).toarray()
    yc = encoder.inverse_transform(yc)
    result.append(yc)
    
print(''.join(result))