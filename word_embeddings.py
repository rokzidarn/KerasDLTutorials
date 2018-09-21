# WORD EMBEDDINGS

from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense

# word embeddings -> dense word vectors
# one-hot vector is sparse (mostly zeros, and big, size of vocabulary), word embeddings are smaller and made of floats
# they pack more info, example: one-hot (20000) vs. word embedding (512)
# word embeddings can be trained with weights on NN or can be added precomputed ones (Word2Vec, GloVe)
# word embeddings differ from taks to task, some work better on sentiment analysis some on QA

# embeddings must have meaning, similar words or even synonyms should be closer to each other
# they should have some sort of geometric representation, distances, vectors for example:
    # dog+wild_vector=wolf, cat+wild_vector=tiger, dog and cat are similar (small distance), both pets

# TODO: 186 + 187

embedding_layer = Embedding(1000, 64)  # (number of possible tokens, dimensionality)
# takes indices of words from dict ({1: "hello", 2: "good", ...}) -> and transforms to dense vectors

max_features = 10000  # 10k most common words
maxlen = 20  # take first 20 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)  # samples, words, list of ints
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)  # 2D integer tensor
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# embedding sentences in sequences of vectors, flattening them, and training a Dense layer

model = Sequential()

model.add(Embedding(10000, 8, input_length=maxlen))  # 8 dimensional embedding
model.add(Flatten())  # 3D -> 2D tensor
model.add(Dense(1, activation='sigmoid'))  # this model treats words as single units, not a group of words

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
history_dict = history.history

print("\n", history_dict['val_acc'][-1])
