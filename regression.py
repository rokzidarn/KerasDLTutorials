from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def build_model():
    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # no activation function, no constraints on predicted values
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

# regression problem: consists of predicting a continuous value instead of a discrete label (i.e. (0,1))
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# 404 examples, 102 tests, 13 features

# normalize data, feature-wise normalization, feature is centered around 0 and has a unit standard deviation (-1,1)
mean = train_data.mean(axis=0)
train_data -= mean
test_data -= mean

std = train_data.std(axis=0)
train_data /= std
test_data /= std

k = 4  # k-fold validation
num_val_samples = len(train_data) // k  # // decimals removed
num_epochs = 100
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(average_mae_history)

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
# MAE is at minimum when 60 epoch is hit after that we are overfitting
