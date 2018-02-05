# -*- coding: utf-8 -*-
from utils import *
from sequentials import get_classifier_1lstm_short_mem, get_classifier_2lstm_medium_mem, get_classifier_1lstm_long_mem
from keras.callbacks import TensorBoard
import math

ARTIST = 'halott-penz'
EPOCHS = 10
PATIENCE_LIMIT = 25
data = read_file('dataset/%s.txt' % ARTIST)
DATA_SLICE = len(data) // 10

log = TrainLogger('train-log-%s.csv' % ARTIST)

TENSOR_LOGGER = './tf-logs'
tensorboard = TensorBoard(TENSOR_LOGGER)
train_log_per_batch_names = ['train_batch_loss', 'train_batch_accuracy']
train_log_per_epoch_names = ['train_epoch_loss', 'train_epoch_accuracy']
val_log_names = ['val_loss', 'val_accuracy']
test_log_names = ['test_loss', 'test_accuracy']

encoder = load_or_create_encoder(ARTIST, ENCODER_FORMAT_LOWERCASE, data)
data = encoder.transform(data)
# Split data for testing and validating purposes
val_data = data[0: DATA_SLICE]
test_data = data[DATA_SLICE:2*DATA_SLICE]
data = data[DATA_SLICE:]
chunk = DataChunk(data, 100, 300)
test_chunk = DataChunk(test_data, 100, 300)

classifier = get_classifier_1lstm_short_mem(*chunk.get_dummy())
tensorboard.set_model(classifier)

min_loss = math.inf
patience = 0
best_model_name = None
global_steps = 0
for epoch in range(EPOCHS):
    print('\n[%d]Epoch %d/%d' % (epoch + 1, epoch + 1, EPOCHS))
    chunk = DataChunk(data, 100, 300)
    val_chunk = DataChunk(val_data, 100, 300)
    losses, accs, v_losses, v_accs = [], [], [], []
    for i, (train_X, train_y) in enumerate(iter(chunk)):
        loss, acc = classifier.train_on_batch(train_X, train_y)
        print('[%d]Batch %d: loss = %f, acc = %f' % (epoch + 1, i + 1, loss, acc))
        write_log_to_board(tensorboard, train_log_per_batch_names, (loss, acc), global_steps)
        losses.append(loss)
        accs.append(acc)
        global_steps += 1

    for (val_X, val_y) in iter(val_chunk):
        val_loss, val_acc = classifier.test_on_batch(val_X, val_y)
        v_losses.append(val_loss)
        v_accs.append(val_acc)

    val_loss_avg = np.average(v_losses)
    val_acc_avg = np.average(v_accs)
    train_loss_avg = np.average(losses)
    train_acc_avg = np.average(accs)
    write_log_to_board(tensorboard, val_log_names, (val_loss_avg, val_acc_avg), global_steps)
    write_log_to_board(tensorboard, train_log_per_epoch_names, (train_loss_avg, train_acc_avg), global_steps)
    print('[%d]FINISHING EPOCH.. val_loss = %f, val_acc = %f' % (epoch + 1, val_loss_avg, val_acc_avg))

    log.add_entry(train_loss_avg, train_acc_avg, val_loss_avg, val_acc_avg)

    if val_loss_avg <= min_loss:
        min_loss = val_loss_avg
        patience = 0
        best_model_name = '%s_e_%d_l_%.4f' % (ARTIST, epoch, min_loss)
        persist_model(classifier, best_model_name)
        print('[%d]New best model for validation set found.. val_loss = %f, val_acc = %f' % (epoch + 1, val_loss_avg, np.average(v_accs)))
    elif patience >= PATIENCE_LIMIT:
        print('[%d]Patience limit (%d) reached stopping iteration. Best validation loss found was: %f' % (epoch + 1, PATIENCE_LIMIT, min_loss))
        break
    else:
        patience += 1

classifier = load_model(best_model_name)
classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])

t_losses, t_accs = [], []
for i, (test_X, test_y) in enumerate(iter(test_chunk)):
    test_loss, test_acc = classifier.test_on_batch(test_X, test_y)
    write_log_to_board(tensorboard, test_log_names, (test_loss, test_acc), i)
    t_losses.append(test_loss)
    t_accs.append(test_acc)

print('Best model\'s test_loss = %f, test_acc = %f' % (np.average(t_losses), np.average(t_accs)))