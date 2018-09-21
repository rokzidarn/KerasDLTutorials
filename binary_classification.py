from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def decode_review(review_index):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[review_index]])
    return decoded_review

def vectorize_sequences(sequences, dimension=10000):
    # transforms to one-hot vector, 10k dimension because of most frequent words
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def plot_loss(history_dict, epochs):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_acc(history_dict, epochs):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# MAIN

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 25k samples on each train/test set
# only 10k most frequent words kept, others (rare) are discarded

#print(train_data[0])  # words used in the review, numbered
#print(train_labels[0])  # class (0,1) -> (negative, positive)
#print(decode_review(0))  # prints review in text format

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')  # vectorize labels
y_test = np.asarray(test_labels).astype('float32')

# Dense(16, activation='relu') -> 16 neurons on layer, input data is transformed into 16-dimensions
# more neurons means the network can learn more complex data representations, but that takes longer
# and can cause overfitting

model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

x_val = x_train[:10000]  # validation set from training data
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

num_of_epochs = 20
history = model.fit(partial_x_train, partial_y_train, epochs=num_of_epochs, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history  # data during training, history_dict.keys()
epochs = range(1, num_of_epochs + 1)

#plot_loss(history_dict, epochs)
#plot_acc(history_dict, epochs)

# plots show that loss and accuracy are at minimum on 4th epoch, which means that they rise after that
# this signifies that we are overfitting on training data, to achieve best results, we only need 4 epochs

results = model.evaluate(x_test, y_test)
model.predict(x_test)
