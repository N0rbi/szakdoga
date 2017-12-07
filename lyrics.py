# -*- coding: utf-8 -*-

import numpy as np
from utils import CharEncoder, read_file

def get_classifier_4lstm_4do(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    classifier = Sequential()
    classifier.add(LSTM(units = 50, return_sequences=True, input_shape=X_train.shape[1:3]))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(LSTM(units = 50, return_sequences=True))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(LSTM(units = 50, return_sequences=True))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(LSTM(units = 50, return_sequences=False))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(Dense(units = y_train.shape[1], activation='softmax'))
    
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy")

    return classifier

def get_classifier_1lstm_long_mem(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    classifier = Sequential()
    classifier.add(LSTM(units = 256, input_shape=X_train.shape[1:3]))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(Dense(units = y_train.shape[1], activation='softmax'))
    
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy")

    return classifier

def get_classifier_1lstm_short_mem(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    classifier = Sequential()
    classifier.add(LSTM(units = 10, input_shape=X_train.shape[1:3]))
    classifier.add(Dropout(rate=0.2))
    
    classifier.add(Dense(units = y_train.shape[1], activation='softmax'))
    
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    
    return classifier

data = read_file('dataset/30y.txt')

encoder = CharEncoder()

data = encoder.fit_transform(data)

train_X = []
train_y = []
for i in range(0, len(data) - 100):
    current_in = data[i:i+100]
    current_out = data[i + 100]
    train_X.append(current_in)
    train_y.append(current_out)

train_X = np.array(train_X)
train_y = np.array(train_y)


classifier = get_classifier_1lstm_long_mem(train_X, train_y)

classifier.fit(train_X, train_y, epochs=60, batch_size=1000)

serialized = classifier.to_json()

with open("model_4.json", "w") as json:
    json.write(serialized)
    
classifier.save_weights("model_4.h5")

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
Nekem senkim, de senkim sincsen

A torkom összeszorul
Járni alig bírok
Az útra napfény borul
Ha rád gondolok, sírok
Nincs már miben hinnem
Ráuntam a tájra
Nekem senkim sincsen
Most látsz utoljára

Param, pararam
Pararara-rararam
Param, pararam
Á-há

(Refrén 2×)

Á-há

(Refrén)

Fáj a fejem, a szívem túl nagy
És nem tudom, nem tudom, hol vagy
Forog a világ, elfolyik minden
Nekem senkim, de senkim, de senkim, de senkim sincsen'''

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