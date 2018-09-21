# Convolutional neural networks (convnets)

# used in computer vision -> image classification
# convnets learn local parameters instead of global
# after a pattern is recognized (in upper left corner) it is also recognized elsewhere (center)
# location is not important as in densely connected layers
# they can also learn to connect these pattern into new, bigger ones

# TODO: page 122 -> 5.1.1 + 5.1.2

from keras import layers
from keras import models

model = models.Sequential()

# learning stands on feature maps (height + width + (depth -> 3 levels (RGB color) or 1 level (BW - grey shades))
# depth = channels
# output depth = filter, not color, but presence of certain features (i.e. face)
# convolution kernel (weight matrix) produces the filter

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# first layers act as a group of edge detectors, result of the network is still recognizable
# next layers are more abstract, detecting certain features (ears, noses), but less recognizable

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# (output depth, (sliding windows dimensions))
model.add(layers.MaxPooling2D((2, 2)))
# downsample feature maps by factor 2, outputs max value of channel
# reduce the number of feature-map coefficients to process,

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Dropout(0.5))
# consists in randomly setting a fraction rate of input units to 0 at each update during training time, less overfitting
model.add(layers.Flatten())
# flattens input, (64*32*32) -> 65536

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # binary classification, predicting

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5, batch_size=64)

model.save('history.h5')  # save model on disk