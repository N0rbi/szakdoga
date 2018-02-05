# -*- coding: utf-8 -*-

import numpy as np
from utils import CharEncoder, read_file, persist_model, load_model, DataChunk, TrainLogger
from sequentials import get_classifier_1lstm_short_mem, get_classifier_2lstm_medium_mem, get_classifier_1lstm_long_mem
import math

ARTIST = 'halott-penz'
EPOCHS = 600
data = read_file('dataset/%s.txt' % ARTIST)
encoder = CharEncoder() #TODO save encoder as well
data = encoder.fit_transform(data)
DATA_SLICE = int(len(data) / 10)
# Split data for testing and validating purposes
val_data = data[0: DATA_SLICE]
test_data = data[DATA_SLICE:2*DATA_SLICE]
data = data[DATA_SLICE:]
chunk = DataChunk(data, 100, 300)
test_chunk = DataChunk(test_data, 100, 300)
classifier = get_classifier_1lstm_long_mem(*chunk.get_dummy())

log = TrainLogger('train-log-%s.csv' % ARTIST)
# define the checkpoint
# filepath=os.path.join("checkpoints", "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=250, write_graph=True, write_images=False)
# callbacks_list = [checkpoint, tensorboard]
min_loss = math.inf
patience = 0
patience_limit = 25
best_model_name = None
for epoch in range(EPOCHS):
    print('\n[%d]Epoch %d/%d' % (epoch + 1, epoch + 1, EPOCHS))
    chunk = DataChunk(data, 100, 300)
    val_chunk = DataChunk(val_data, 100, 300)
    losses, accs, v_losses, v_accs = [], [], [], []
    for i, (train_X, train_y) in enumerate(iter(chunk)):
        loss, acc = classifier.train_on_batch(train_X, train_y)
        print('[%d]Batch %d: loss = %f, acc = %f' % (epoch + 1, i + 1, loss, acc))
        losses.append(loss)
        accs.append(acc)

    for (val_X, val_y) in iter(val_chunk):
        val_loss, val_acc = classifier.test_on_batch(val_X, val_y)
        v_losses.append(val_loss)
        v_accs.append(val_acc)

    val_loss_avg = np.average(v_losses)
    print('[%d]FINISHING EPOCH.. val_loss = %f, val_acc = %f' % (epoch + 1, val_loss_avg, np.average(v_accs)))

    log.add_entry(np.average(losses), np.average(accs), val_loss_avg, np.average(v_accs))

    if val_loss_avg <= min_loss:
        min_loss = val_loss_avg
        patience = 0
        best_model_name = '%s_e_%d_l_%.4f' % (ARTIST, epoch, min_loss)
        persist_model(classifier, best_model_name)
        print('[%d]New best model for validation set found.. val_loss = %f, val_acc = %f' % (epoch + 1, val_loss_avg, np.average(v_accs)))
    elif patience >= patience_limit:
        print('[%d]Patience limit (%d) reached stopping iteration. Best validation loss found was: %f' % (epoch + 1, patience_limit, min_loss))
        break
    else:
        patience += 1

classifier = load_model(best_model_name)
classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])

t_losses, t_accs = [], []
for (test_X, test_y) in iter(test_chunk):
    test_loss, test_acc = classifier.test_on_batch(test_X, test_y)
    t_losses.append(test_loss)
    t_accs.append(test_acc)

print('Best model\'s test_loss = %f, test_acc = %f' % (np.average(t_losses), np.average(t_accs)))