# -*- coding: utf-8 -*-
from utils import *
from sequentials import *


def train(artist, epochs, patience_limit, lstm_layers, lstm_units, embedding, size_x, model_name):
    from sequentials import get_classifier
    from keras.callbacks import TensorBoard, EarlyStopping

    if not artist:
        print('You need to pick an artist first')
        exit(-1)

    BATCH_SIZE=32
    data = read_file('dataset/%s.txt' % artist)
    DATA_SLICE = len(data) // 10
    artifact_params = (artist, size_x) if not model_name else (artist, size_x, model_name)
    artifact = ModelArtifact(*artifact_params)
    tensor_logger = artifact.get_tensorflow_logdir()
    encoder = artifact.load_or_create_encoder(data)

    os.makedirs(artifact.get_tensorflow_logdir(), exist_ok=True)
    metadata_file_name = os.path.abspath(os.path.join(artifact.get_tensorflow_logdir(), "metadata" + ".tsv"))
    save_metadata_of_embedding(metadata_file_name, encoder.vocab)

    tensorboard = TensorBoard(tensor_logger, embeddings_metadata=metadata_file_name, embeddings_freq=30)
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=patience_limit, verbose=0, mode='auto')

    # Split data for testing and validating purposes
    val_data = encoder.transform(data[0: DATA_SLICE], with_onehot=False)
    test_data = encoder.transform(data[DATA_SLICE:2*DATA_SLICE], with_onehot=False)
    data = encoder.transform(data[DATA_SLICE:], with_onehot=False)
    classifier = get_multitask_classifier(BATCH_SIZE, size_x, len(encoder.vocab), 4, lstm_layers, embedding, lstm_units)

    callbacks = [tensorboard, earlyStop]

    classifier.fit_generator(read_batches(data, len(encoder.vocab), BATCH_SIZE, size_x, epochs, encoder),
                             steps_per_epoch=int(data.shape[0]/size_x / BATCH_SIZE),
                             epochs=epochs,
                             callbacks=callbacks,
                             validation_data=read_batches(val_data, len(encoder.vocab), BATCH_SIZE, size_x, epochs, encoder),
                             validation_steps=1,
                             )

    classifier.evaluate_generator(read_batches(test_data, len(encoder.vocab), BATCH_SIZE, size_x, 1, encoder),
                                  steps=int(test_data.shape[0]/ size_x / BATCH_SIZE))


def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Lyrics generating with Keras and recurrent networks')
    parser.add_argument('--artist', type=str, help='The dataset to be used during learning.')
    parser.add_argument('--epochs', type=int, help='For how many epochs the program should learn.', default=250)
    parser.add_argument('--patience_limit', type=int,
                        help='At which epoch after not increasing accuracy the program should terminate.', default=25)
    parser.add_argument('--lstm_layers', type=int, help='How many layers of lstm should the model be built with.',
                        default=3)
    parser.add_argument('--lstm_units', type=int, help='How many hidden units the lstm layers should have.', default=64)
    parser.add_argument('--embedding', type=int, help='How many dimensions should the embedding project to.',
                        default=500)
    parser.add_argument('--size_x', type=int, help='How long should the the input be.', default=100)
    parser.add_argument('--model_name', type=str,
                        help='Name of the model (if not given it will use the timestamp followed by the artist).',
                        default=None)

    args = vars(parser.parse_args())

    train(**args)


if __name__ == '__main__':
    cli()
