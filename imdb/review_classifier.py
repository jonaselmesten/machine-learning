import keras
from keras import models, layers, regularizers
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences

from visualize import plot_history

callbacks_list = [keras.callbacks.ModelCheckpoint(filepath="models/epoch_{epoch:02d}-val_loss_{val_loss:.2f}-val_acc_{val_acc:.2f}.h5",
                                                  save_freq="epoch",
                                                  save_best_only=True,
                                                  save_weights_only=False),
                  keras.callbacks.EarlyStopping(monitor="val_loss",
                                                mode="auto",
                                                restore_best_weights=True,
                                                patience=2),
                  keras.callbacks.TensorBoard(log_dir="logs")]

max_words = 10000
max_len = 30

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)
word_index = imdb.get_word_index()

train_data = pad_sequences(train_data, maxlen=max_len)
test_data = pad_sequences(test_data, maxlen=max_len)

model = models.Sequential()
model.add(layers.Embedding(max_words, 8, input_length=max_len))
model.add(layers.LSTM(4, return_sequences=True))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(train_data, train_labels,
                    callbacks=callbacks_list,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,)

plot_history(history)

# When testing - layer_output *= 0.5 if 0.5 dropout was used during training.
