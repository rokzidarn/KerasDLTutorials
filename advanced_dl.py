# ADVANCED DEEP LEARNING

from keras.models import Sequential, Model
from keras import layers
from keras import Input
import numpy as np

# batch normalization, normalization is usually applied before learning (data is centered around 0, with std 1)
# example: normalized_data = (data - np.mean(data)) / np.std(data)
# data can be changed during learning, but we can also apply normalization during learning
# example: model.add(layers.BatchNormalization())  # apply after Dense layer

# hyperparameter optimization, hyperparameters are #layers, #filters/layer, type of normalization, type of AF
# no recipe, just trial and error, plus develop fine tunning skilss over time
# Keras hyper optimization library: https://github.com/maxpumperla/hyperas

# ensembling, multiple good models, that are avereged, by using weights depending on how good they are
# some work good on certain samples, some on other, all work similarly good, together they work the best
# the goal is that they work as best as possible individually, but also differently from eachother
# example:
    #preds_a = model_a.predict(x_val)
    #preds_b = model_b.predict(x_val)
    #preds_c = model_c.predict(x_val)
    #preds_d = model_d.predict(x_val)
    #final_preds = 0.5 * preds_a + 0.25 * preds_b + 0.1 * preds_c + 0.15 * preds_d

# Keras functional API
# multimodal inputs: they merge data coming from different input sources, processing each type of data using
    # different kinds of neural layers
# example: Dense + RNN + ConvNet -> Merging module

seq_model = Sequential()  # classic implementation
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

input_tensor = Input(shape=(64,))  # functional implementation
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)

# dimensionality of the word2vec embedding space is usually lower than the dimensionality of the one-hot
    # embedding space, which is the size of the vocabulary, the embedding space is also more dense compared
    # to the sparse embedding of the one-hot embedding space, implemented by CBOW or SkipGram

# example: I love green eggs and ham -> ([I, green], love), ([love, eggs], green)

# word2vec is a predictive model (10 billion words)
# GloVe is a count-based model (matrix factorization + SGD), higher accuracy + faster learning (8 billion words)