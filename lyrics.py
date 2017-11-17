# -*- coding: utf-8 -*-

import numpy as np

class RNNVocabData():
    
    def __init__(self, file_name, seq_len, preprocess_X, preprocess_y):
        self.__file_name = file_name
        self.__seq_len = seq_len
        self.__file_content = None
        self.__preprocess_X = preprocess_X
        self.__preprocess_y = preprocess_y
        
    def __read_file(self):
        if not self.__file_content:
            with open(self.__file_name, encoding='utf-8') as file_ref:
                content = file_ref.read()
                #we only want to learn the words not the capitation
                self.__file_content = content.lower()
        content = self.__file_content
        content = self.__create_vocab(content)
        return content
    
    # Enumerates the used vocablurary and assigns an id to each element
    def __create_vocab(self, content):
        vocab = sorted(list(set(content)))
        return dict((c,i) for i, c in enumerate(vocab))
    
    def get_data_set(self):
        vocab = self.__read_file()
        train_X = []
        train_y = []
        for i in range(0, len(self.__file_content) - self.__seq_len):
            current_in = self.__file_content[i:i+self.__seq_len]
            current_out = self.__file_content[i + self.__seq_len]
            train_X.append([vocab[ch] for ch in current_in])
            train_y.append(vocab[current_out])
        
        train_X = np.array(train_X)
        train_y = np.array(train_y)
            
        if self.__preprocess_X:
            train_X = self.__preprocess_X(train_X)
        
        if self.__preprocess_y:
            train_y = self.__preprocess_y(train_y)
        
        train_X = np.reshape(train_X, train_X.shape+(1,))
        
        return train_X , train_y
    
    def format_string(self, string):
        vocab = self.__read_file()
        result = []
        for i in range(0, len(string) - self.__seq_len):
            current_in = string[i:i+self.__seq_len]
            result.append([vocab[ch] for ch in current_in])
        return result
        
    def decode_string(self, encoded):
        rev_vocab = {v: k for k, v in self.__read_file().items()}
        return [rev_vocab[key] for key in encoded]

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

from sklearn.preprocessing import Normalizer, OneHotEncoder

normalizer_X = Normalizer()
one_hot_y = OneHotEncoder()


def preprocess_X(X_train):
    return normalizer_X.fit_transform(X_train)

def preprocess_y(y_train):
    y_train = y_train.reshape(-1, 1)
    return one_hot_y.fit_transform(y_train).toarray()

data = RNNVocabData('dataset/30y.txt', 100, preprocess_X, preprocess_y)

X_train, y_train = data.get_data_set()

classifier = get_classifier_1lstm_long_mem(X_train, y_train)

classifier.fit(X_train, y_train, epochs=100, batch_size=25)


serialized = classifier.to_json()

with open("model_1.json", "w") as json:
    json.write(serialized)
    
classifier.save_weights("model_1.h5")

'''
sample = "Csak most ne, csak most ne,"
sample = np.array(list(map(ord, sample)))
sample = sample.reshape(sample.shape+(1,))

sample = scaler.transform(sample)

sample = sample.reshape(sample.shape+(1,))


X_test = []

for i in range(5, sample.size-1):
    X_test.append(sample[i-15:i, 0])
    
X_test = np.array(X_test)

predicted = classifier.predict(X_test)

predicted = scaler.inverse_transform(predicted)

predicted = predicted.astype(int).flatten().tolist()

string = [chr(x) for x in predicted]

print(string)
''' 

#Load the content

from keras.models import model_from_json
loaded = open('model.json', 'r')
model = loaded.read()
loaded.close()
model = model_from_json(model)
model.load_weights("model.h5")

model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

sample = '''hogyha az életben 
boldog akarsz lenni
csinálj zenekart és hogyha
abból nem lesz semmi
vegyél magad mellé egy
kutyát vagy macskát társnak 
mert nem beszél vissza egy
ilyen háziállat

látni a szíveddel
a szemeddel lesni
ha becsukod vágyak
és ha, ha kinyitod semmi
majd a hintaszéked
elringat s az álom
nem jön a szemedre csak
ott ül a párkányon

ha feldobod, piros
ha leesik, semmi
ha az élet fenn van'''


sample = data.format_string(sample)
sample = np.array(sample)
sample = normalizer_X.transform(sample)
sample = sample.reshape(sample.shape + (1,))


predicted = model.predict(sample)

seq = []
for prediction in predicted:
    seq.append(prediction.argmax())
    
seq = data.decode_string(seq)
seq = ''.join(seq)