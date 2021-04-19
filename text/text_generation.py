import numpy as np


def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)

text = open("files/nietzsche.txt").read().lower()
print("Total length:", len(text))

max_len = 60
step = 3

sentences = []
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])

print("Num of sequences:", len(sentences))

print(sentences[4])
print(next_chars[4])