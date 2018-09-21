# RECURRENT NETWORK

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
from keras.preprocessing import sequence
from keras.layers import Dense
import matplotlib.pyplot as plt

def simple_rnn():
    timesteps = 100
    input_features = 32
    output_features = 64

    inputs = np.random.random((timesteps, input_features))  # data
    state_t = np.zeros((output_features,))  # init of first state, because it is not defined in data

    W = np.random.random((output_features, input_features))  # weights
    U = np.random.random((output_features, output_features))
    b = np.random.random((output_features,))  # bias

    successive_outputs = []
    for input_t in inputs:
        output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)  # output dependent on input and previous state
        successive_outputs.append(output_t)  # results of processing
        state_t = output_t  # new current state

    return np.concatenate(successive_outputs, axis=0)

# usual networks, feedforward networks have no memory, everything is processed in a single sequence
    # for example, a whole movie review is transformed into a vector
# recurrent neural networks a sequence is processed by elements while maintaining a state which is containing
    # information what has already been seen, learned and is updated by new information that is next
    # internal loop, new state is dependant on previous state and new element of the sequence
    # but when a new sample occurs, the data is reset
# with RNNs you have backpropagation through time, gradient depends not only on current input but also previous ones

# TODO: 198

# CLASSIFICATION EXAMPLE

max_features = 10000
maxlen = 500
batch_size = 32

# prepare data
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

# define model
model = Sequential()
model.add(Embedding(max_features, 32))

# if you stack multiple SimpleRNN layers, all but last one must return full outputs
# SimpleRNN(32, return_sequences=True)
# only the last has to return the last output
model.add(SimpleRNN(32))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# plot results
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

# basic recurrent networks are bad on longer sequences such as text, because they are too simplistic, LSTM is better
