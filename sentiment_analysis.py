# SENTIMENT ANALYSIS
# pretrained word embeddings + raw text data from IMDB

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt

# 1. GETTING RAW TEXT WITH LABELS

imdb_dir = 'imdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []  # label of each review
texts = []  # single review

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname, ), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# 2. TOKENIZING THE RAW TEXT

maxlen = 100  # first 100 words of the review
training_samples = 20000  # training done on 2000 samples, out of 25k samples
validation_samples = 5000
max_words = 10000  # 10k most common words

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)  # words represented as a number i.e. "home" = 2
word_index = tokenizer.word_index  # dictionary of distinct words -> (key, value) - ('word', index)
sequences = tokenizer.texts_to_sequences(texts)  # each review represented as numerical array,
# i.e. "hello sweet home..." = [33, 199, 2, ...]

data = pad_sequences(sequences, maxlen=maxlen)  # pads sequences to the same length
labels = np.asarray(labels)

#print('Shape of data tensor:', data.shape)  # 25k reviews consisting of first 100 words
#print(word_index["story"])  # value of word, index
#print(data[0])  # first review

# 3. SPLITTING DATA FOR LEARNING

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
print(y_val[0:20]);exit(0)

# 4. GLOVE EMBEDDINGS

glove_dir = 'glove'
embeddings_index = {}  # 400k pretrained word embeddings
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")
# (word, vector representation of the word) -> ('the', [0.26, -0.00031, 0.6, 0.777 ...])

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# 5. ASSOCIATING WORDS WITH GLOVE VECTOR REPRESENTATIONS

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():  # goes through all the words in the reviews, each word represented by unique number
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:  # if the word is found in the pretrained embeddings, get vector for the word
            embedding_matrix[i] = embedding_vector
        # words not found in the embedding index will be all zeros

#print(embedding_matrix[1])  # glove vector of word at index 1, from word_index

# the process: word_index ('the': 121), embedding_index (121. index of array -> [0.001, 0.9312, -0.3, ...])

# 6. DEFINING THE MODEL

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# has single weight matrix, 2D, each word has a vector for weights, this is usually trained, not in this case

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])  # set the pretrained glove weights
model.layers[0].trainable = False

# 7. TRAINING WITH GLOVE
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# 8. TRAINING WITHOUT GLOVE

"""
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
"""

# 9. EVALUATION

test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.h5')
results = model.evaluate(x_test, y_test)
print(model.metrics_names)  # ['loss', 'acc']
print("acc:", results[1])
