# -*- coding: utf-8 -*-

import numpy as np

def read_file(file, separator):
    with open(file, encoding='utf-8') as file_ref:
        content = file_ref.read()
        if separator:
            content = content.split(separator)
            for i in range(len(content)):
                content[i] = list(map(ord, content[i]))
            content = np.array(content)
        else:
            content = list(map(ord, content))
            content = np.array(content)
            content = content.reshape(content.shape+(1,))
    return content

def read_by_letters(file):
    return read_file(file, None)

def read_by_words(file):
    return read_file(file, ' ')

def read_by_line(file):
    return read_file(file, '\n')

def generate_test_data(data, prev, out):
    X_train = []
    y_train = []
    
    for i in range(prev, data.size-out):
        X_train.append(data[i-prev:i, 0])
        y_train.append(data[i:i+1, 0])
    
    return np.array(X_train), np.array(y_train)


def get_classifier(X_train):
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
    
    classifier.add(Dense(units = 1))
    
    classifier.compile(optimizer="adam", loss="binary_crossentropy")

    return classifier

letters = read_by_letters('dataset/30y.txt')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
letters = scaler.fit_transform(letters)

X_train, y_train = generate_test_data(letters, 15, 1)

# No correlated data
X_train = X_train.reshape(X_train.shape+(1,))

classifier = get_classifier(X_train)

classifier.fit(X_train, y_train, epochs=50, batch_size=25)

sample = "Csak most ne, csak most ne,"
sample = np.array(list(map(ord, sample)))
sample = sample.reshape(sample.shape+(1,))

sample = scaler.transform(sample)

sample = sample.reshape(sample.shape+(1,))


X_test = []

for i in range(15, sample.size-1):
    X_test.append(sample[i-15:i, 0])
    
X_test = np.array(X_test)

predicted = classifier.predict(X_test)

predicted = scaler.inverse_transform(predicted)

predicted = predicted.astype(int).flatten().tolist()

string = [chr(x) for x in predicted]

print(string)

