# TEXT PROCESSING

import numpy as np
from keras.preprocessing.text import Tokenizer

# text -> sequence data (words)
# examples: document classification, sentiment analysis, question answering
# pattern recognition applied to words, sentences and paragraphs
# tokenization
# text vectorizing transforms text into numeric tensors
# 2 approaches: one-hot encoding and word embeddings (token embedding)
# N-gram = sequence of N consecutive words

# ONE HOT ENCODING
# vector of a word consists of all zeros except one 1 -> [0 0 0 0 0 1 0 0 0 ]
# length of the vector is the size of the vocabulary, number of words in the text

# TEXT CONVOLUTION
# 1D convolution can be used in sequence processing, instead of 2d tensors in images, we can use 1D tensor in text
# we have some sort of input text, and a sliding windows of certain width, this can recognize local patterns
# Conv1D, also used because of faster computation


def one_hot_encoding(samples):
    token_index = {}

    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1  # no all zero vector

    max_length = 10
    results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

    return results

# MAIN

text = ['The cat sat on the mat.', 'The dog ate my homework.']
r = one_hot_encoding(text)
#print(r)

# Keras implementation of one-hot encoding
tokenizer = Tokenizer(num_words=1000)  # number of most common words, length of one-hot vector
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
one_hot_results = tokenizer.texts_to_matrix(text, mode='binary')
word_index = tokenizer.word_index  # dictionary of all words in text
print(word_index)
# if the number of unique tokens in your vocabulary is too large to handle use hashin trick