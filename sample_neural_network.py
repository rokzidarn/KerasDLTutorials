from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)  # (#samples, width, height)
print(len(train_labels))  # number of training samples

print(test_images.shape)  # (#samples, width, height)
print(len(test_labels))  # number of testing samples

network = models.Sequential()

# each layer extracts a representations from input, chaining layers forms "data distillation"
# dense = fully connected, each neuron
# input = image dimension
network.add(layers.Dense(512, activation='relu', input_shape=(train_images.shape[1] * train_images.shape[2],)))  # 28x28
network.add(layers.Dense(10, activation='softmax'))  # output = 10 classes, numbers from 0...9

# (optimizer = weight updater, loss = loss function, performance measurement, metric of success)
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# data preparation, color: [0,255] -> [0,1]
train_images = train_images.reshape((60000, 28 * 28))  # (#samples, width * height)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

print(train_labels[0])
train_labels = to_categorical(train_labels)  # class (number 5) -> one hot vector [0,0,0,0,0,1,0,0,0,0]
print(train_labels[0])
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# 2D (samples, features) -> Dense
# 3D (samples, timestamps, features) -> LSTM
# 4D (images) -> Conv2D
