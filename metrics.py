import keras.backend as K


def perplexity(y_true, y_pred):
    crossentropy = K.categorical_crossentropy(y_true, y_pred)
    return K.pow(2.0, crossentropy)
