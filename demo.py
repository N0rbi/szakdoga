from utils import load_model, read_file, CharEncoder
import numpy as np

ARTIST = 'halott-penz'

classifier = load_model('%s_e_10_l_2.4009' % ARTIST)
t_losses, t_accs = [], []
classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy")

sample = '''A buli íze a számban
Lányok neonruhában
Oldódnak a színpadon
Részeg vagyok, nem is tudom
Mire jó, ha jó ez'''

data = read_file('dataset/%s.txt' % ARTIST)
encoder = CharEncoder() #TODO save encoder as well
data = encoder.fit(data)

sample = encoder.transform(sample)


def predict_next(text):
    x = text[-100:]
    x = np.array([x])
    probabilities = classifier.predict(x)
    y = np.array([np.random.choice(len(prob), p=prob) for prob in probabilities])
    y = encoder.onehot.transform(np.array(y).reshape(-1, 1)).toarray()
    text = np.append(text, y, axis=0)
    return text


def predict_next_n(text, n):
    if n == 0:
        return text
    else:
        return predict_next_n(predict_next(text), n - 1)


prediction = predict_next_n(sample, 400)

result = []

for yc in prediction:
    yc = yc.reshape(1, -1)
    yc = encoder.inverse_transform(yc)
    result.append(yc)

print(''.join(result))
