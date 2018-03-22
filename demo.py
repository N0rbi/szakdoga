from utils import load_model, load_or_create_encoder, ENCODER_FORMAT_LOWERCASE
import numpy as np

ARTIST = '30y'

classifier = load_model('30y_e_9_l_2.1595', from_reference=True)
t_losses, t_accs = [], []
classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

sample = '''A buli íze a számban
Lányok neonruhában
Oldódnak a színpadon
Részeg vagyok, nem is tudom
Mire jó, ha jó ez'''

encoder = load_or_create_encoder(ARTIST, ENCODER_FORMAT_LOWERCASE, None)

sample = encoder.transform(sample, False)


def predict_next(text):
    x = text[-100:]
    x = np.array([x])
    probabilities = classifier.predict(x)
    y = np.array([np.random.choice(len(prob), p=prob) for prob in probabilities])
    text = np.append(text, y)
    return text


def predict_next_n(text, n):
    if n == 0:
        return text
    else:
        return predict_next_n(predict_next(text), n - 1)


prediction = predict_next_n(sample, 300)

result = encoder.inverse_transform(prediction)


print(''.join(result))
