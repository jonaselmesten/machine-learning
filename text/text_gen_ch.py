import random

import keras
import numpy as np
from keras import layers
from tqdm import tqdm

callbacks_list = [keras.callbacks.ModelCheckpoint(filepath="models/epoch_{epoch:02d}.h5",
                                                  save_freq="epoch",
                                                  period=5,
                                                  save_best_only=False,
                                                  save_weights_only=False)]

# TODO: Try this but with Embedding layer instead of one-hot encoding.

text = open("files/nietzsche.txt").read().lower()

max_len = 60
step = 3

sentences = []
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])

unique_chars = sorted(list(set(text)))
char_indices = dict((char, unique_chars.index(char)) for char in unique_chars)

print("Total length:", len(text))
print("Unique chars:", len(unique_chars))
print("Num of sequences:", len(sentences))


def one_hot_encode():
    x = np.zeros((len(sentences), max_len, len(unique_chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(unique_chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return x, y


def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


sequences, targets = one_hot_encode()


class TextGen:

    def __init__(self):
        self.model = keras.models.Sequential()
        self.model.add(layers.LSTM(128, input_shape=(max_len, len(unique_chars))))
        self.model.add(layers.Dense(len(unique_chars), activation="softmax"))

        optimizer = keras.optimizers.RMSprop(lr=0.01)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    def load_model(self, model_path=None):
        self.model.load_weights("models/epoch_05.h5")

    def train(self):
        self.model.fit(sequences,
                       targets,
                       batch_size=128,
                       epochs=60,
                       callbacks=callbacks_list)

    def sample(self, prediction, temperature=1.0):
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)
        probabilities = np.random.multinomial(1, prediction, 1)

        return np.argmax(probabilities)

    def print_sentence(self, seed_text=None, sentence_len=50, temp=1.0):
        start_index = random.randint(0, len(text) - max_len - 1)

        if seed_text is None:
            generated_text = text[start_index: start_index + max_len]
        else:
            generated_text = seed_text

        result = generated_text
        print("Creating sentence with seed:", generated_text)

        for _ in tqdm(range(sentence_len)):

            sampled = np.zeros((1, max_len, len(unique_chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            prediction = self.model.predict(sampled, verbose=0)[0]
            next_index = self.sample(prediction, temp)
            next_char = unique_chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            result += next_char

        return result


text_gen = TextGen()
text_gen.load_model()
sentence = text_gen.print_sentence()
print(sentence)