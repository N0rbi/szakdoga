from utils import ModelArtifact


def predict(model_name, sample_data):
    from metrics import perplexity
    import numpy as np
    artifact = ModelArtifact()
    classifier = artifact.load_model(model_name)
    # weights = classifier.get_weights()
    # classifier = classifier(batch_size=1)
    # classifier.set_weights(weights)
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", perplexity])
    sample = sample_data if sample_data else \
        '''A buli íze a számban
Lányok neonruhában
Oldódnak a színpadon
Részeg vagyok, nem is tudom
Mire jó, ha jó ez'''

    encoder = artifact.load_or_create_encoder(None, model_name)

    sample = encoder.transform(sample, False)

    def predict_next(text):
        x = text[-100:]
        #hack for not working batch_size
        x = [x for i in range(32)]
        x = np.array([x])[0]
        prob = classifier.predict(x)[0][-1]

        y = np.random.choice(len(prob), p=prob)

        text = np.append(text, y)
        return text

    def predict_next_n(text, n):
        if n == 0:
            return text
        else:
            return predict_next_n(predict_next(text), n - 1)

    prediction = predict_next_n(sample, 300)

    result = encoder.inverse_transform(prediction)

    return ''.join(result)


def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Testing already learnt models in real life scenarios.')
    parser.add_argument('--model_name', type=str, help='The name of the artifact to load.', default=None)
    parser.add_argument('--sample_data', type=str, help='The first 100 char to generate text uppon.', default=None)

    args = vars(parser.parse_args())

    if not args['model_name']:
        model_names = ModelArtifact.show_all_artifacts()
        print("Choose a model:")
        for i, name in enumerate(model_names):
            print("[%i] %s"% (i, name))

        choosen = None
        while choosen is None:
            try:
                choosen = int(input('Number of the model: '))
            except:
                print("You can only pass numbers ranging from 0 to %i." % (len(model_names)-1))
                pass

        args['model_name'] = model_names[choosen]

    print(predict(**args))


if __name__ == '__main__':
    cli()
