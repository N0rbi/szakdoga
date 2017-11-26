
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
sample = sample.reshape(-1,1)
sample = one_hot_x.transform(sample).toarray()
sample = sample.reshape(sample.shape + (1,))


predicted = model.predict(sample)

seq = []
for prediction in predicted:
    seq.append(prediction.argmax())
    
seq = data.decode_string(seq)
seq = ''.join(seq)