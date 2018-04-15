# -*- coding: utf-8 -*-
from utils import *
from sequentials import *


def train(artist, epochs, patience_limit, lstm_layers, lstm_units, embedding, size_x, model_name):
    from keras.callbacks import TensorBoard
    from metrics import perplexity
    import math

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

    # embeddings_freq=True is a hack for embeddings to be shown
    tensorboard = TensorBoard(tensor_logger, embeddings_metadata=metadata_file_name, embeddings_freq=True)
    # we need the callback to init the visualizer
    train_log_per_batch_names = ['train_batch_loss', 'train_batch_accuracy', 'train_batch_perplexity']
    train_log_per_epoch_names = ['train_epoch_loss', 'train_epoch_accuracy', 'train_epoch_perplexity']
    val_log_names = ['val_loss', 'val_accuracy', 'val_perplexity']
    test_log_names = ['test_loss', 'test_accuracy']

    # Split data for testing and validating purposes
    val_data = encoder.transform(data[0: DATA_SLICE], with_onehot=False)
    test_data = encoder.transform(data[DATA_SLICE:2*DATA_SLICE], with_onehot=False)
    data = encoder.transform(data[DATA_SLICE:], with_onehot=False)
    classifier = get_classifier(BATCH_SIZE, size_x, len(encoder.vocab), lstm_layers, embedding, lstm_units)
    tensorboard.set_model(classifier)

    min_loss = math.inf
    patience = 0
    global_steps = 0
    for epoch in range(epochs):
        print('\n[%d]Epoch %d/%d' % (epoch + 1, epoch + 1, epochs))
        losses, accs, perps, v_losses, v_accs, v_perps = [], [], [], [], [], []

        for i, (X, Y) in enumerate(read_batches(data, len(encoder.vocab), BATCH_SIZE, size_x)):
            loss, acc, perp = classifier.train_on_batch(X, Y)
            print('[%d]Batch %d: loss = %f, acc = %f, perp = %f' % (epoch + 1, i + 1, loss, acc, perp))
            write_log_to_board(tensorboard, train_log_per_batch_names, (loss, acc, perp), global_steps)
            losses.append(loss)
            accs.append(acc)
            perps.append(perp)
            global_steps += 1

        for (val_X, val_y) in read_batches(val_data, len(encoder.vocab), BATCH_SIZE, size_x):
            val_loss, val_acc, v_perp = classifier.test_on_batch(val_X, val_y)
            v_losses.append(val_loss)
            v_accs.append(val_acc)
            v_perps.append(v_perp)
        val_loss_avg = np.average(v_losses)
        val_acc_avg = np.average(v_accs)
        val_perp_avg = np.average(v_perps)
        train_loss_avg = np.average(losses)
        train_acc_avg = np.average(accs)
        train_perp_avg = np.average(perps)
        write_log_to_board(tensorboard, val_log_names, (val_loss_avg, val_acc_avg, val_perp_avg), global_steps)
        write_log_to_board(tensorboard, train_log_per_epoch_names,
                           (train_loss_avg, train_acc_avg, train_perp_avg), global_steps)
        if epoch % 20 == 0:
            save_embedding_to_board(tensorboard.embeddings_ckpt_path, epoch)
        print('[%d]FINISHING EPOCH.. val_loss = %f, val_acc = %f, val_perplexity = %f' %
              (epoch + 1, val_loss_avg, val_acc_avg, val_perp_avg))

        if val_loss_avg <= min_loss:
            min_loss = val_loss_avg
            patience = 0
            artifact.persist_model(classifier)
            print('[%d]New best model for validation set found.. val_loss = %f, val_acc = %f' %
                  (epoch + 1, val_loss_avg, np.average(v_accs)))
        elif patience >= patience_limit:
            print('[%d]Patience limit (%d) reached stopping iteration. Best validation loss found was: %f' %
                  (epoch + 1, patience_limit, min_loss))
            break
        else:
            patience += 1

    classifier = artifact.load_model()
    classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy', perplexity])

    t_losses, t_accs = [], []
    for i, (test_X, test_y) in enumerate(read_batches(test_data, len(encoder.vocab), BATCH_SIZE, size_x)):
        test_loss, test_acc, _ = classifier.test_on_batch(test_X, test_y)
        write_log_to_board(tensorboard, test_log_names, (test_loss, test_acc), global_steps+i)
        t_losses.append(test_loss)
        t_accs.append(test_acc)

    print('Best model\'s test_loss = %f, test_acc = %f' % (np.average(t_losses), np.average(t_accs)))


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
                        default=32)
    parser.add_argument('--size_x', type=int, help='How long should the the input be.', default=100)
    parser.add_argument('--model_name', type=str,
                        help='Name of the model (if not given it will use the timestamp followed by the artist).',
                        default=None)

    args = vars(parser.parse_args())

    train(**args)


if __name__ == '__main__':
    cli()
