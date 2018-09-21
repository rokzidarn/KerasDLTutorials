# ADVANCED RNN
# weather forecasting problem (timeseries) - regression

import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback

            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return np.mean(batch_maes)

def evaluate_basic_ml(lookback, step, float_data, train_gen, val_gen, val_steps):
    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(
        train_gen, steps_per_epoch=500, epochs=10, validation_data=val_gen, validation_steps=val_steps)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# MAIN

# read data
data_dir = 'climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname, encoding="utf8")
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')  # 14 attributes
lines = lines[1:]  # 420k datapoints, samples, timesteps

float_data = np.zeros((len(lines), len(header) - 1))  # store data in numpy array
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

temp = float_data[:, 1]  # temperature plot, data recorded every 10 mins for several years
limit = int(3600/24)*10  # limit plot to 10 days
#plt.plot(range(limit), temp[:limit])
#plt.show()

# preparing data

# lookback = 720, observations will go back 5 days
# steps = 6, observations will be sampled at one data point per hour
# delay = 144, targets will be 24 hours in the future

# preprocess data, use mean and std to normalize, each attribute separately
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# preparing generators for training, validating, testing
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(
    float_data,lookback=lookback,delay=delay,min_index=0,max_index=200000,shuffle=True,step=step,batch_size=batch_size)
val_gen = generator(
    float_data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,step=step,batch_size=batch_size)
test_gen = generator(
    float_data,lookback=lookback,delay=delay,min_index=300001,max_index=None,step=step,batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)

# non-machine learning baseline
#mae = evaluate_naive_method()  # mae, minimal square error
#print(mae*std[1])  # predicted temparature

# basic machine learning baseline
#evaluate_basic_ml(lookback, step, float_data, train_gen, val_gen, val_steps)

# regularization for reccurent networks

# TODO: 216-219

# 1. DROPOUT
# using the same dropout mask at every timestep allows the network to properly propagate its learning error through
    # time, a temporally random dropout mask would disrupt this error signal and be harmful to the learning process

model = Sequential()
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(
    train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

# 2. STACKING LAYERS
# it is generally a good idea to increase the capacity of your network, until overfitting becomes the primary obstacle
# as long as you aren’t overfitting too badly, you’re likely under capacity

model = Sequential()
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

# 3. BIDIRECTIONAL RNN
# RNNs are notably order dependent, they process the timesteps of their input sequences in order,
    #  and shuffling or reversing the timesteps can completely change the representations
    # this is precisely the reason they perform well on problems where order is meaningful

model = Sequential()
model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)