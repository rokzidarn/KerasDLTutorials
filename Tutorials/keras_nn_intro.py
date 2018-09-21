# KERAS NEURAL NETWORK INTRODUCTION TUTORIAL

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility, stochastic process -> random numbers
numpy.random.seed(7)

# binary classification problem (T/F on diabetes)

# 1. Load data
dataset = numpy.loadtxt("Data/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X_train = dataset[0:750, 0:8]
Y_train = dataset[0:750, 8]

# models in Keras are defined as a sequence of layers
# more layers, more complex network topology
# dense class -> fully connected layers

# input dimension = #features
# distribution of weights: uniform = [0, 0.05], normal = gaussian

# 2. Define model
model = Sequential()  # 2 hidden layers + output layer
model.add(Dense(12, input_dim=8, activation='relu'))  # Dense(#neurons, #features, AF)
model.add(Dense(8, activation='relu'))  # rectifier, max(0,x)
model.add(Dense(1, activation='sigmoid'))  # better for binary classification, easier mapping to prediction

# 3. Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # learning process

# loss function to evaluate a set of weights (logarithmic loss) -> binary cross entropy
# optimizer is used to find minimum of loss function, gradient descent -> adam

# 4. Fit model
model.fit(X_train, Y_train, epochs=150, batch_size=10)  # training

# batch size defines the number of instances that are evaluated before a weight update in the network is performed

# 5. Evaluate model
scores = model.evaluate(X_train, Y_train)
print("\nLEARNING %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))

# 6. Predict
X_test = dataset[750:, 0:8]  # 18 examples (768 - 750)
Y_test = dataset[750:, 8]

predictions = model.predict(X_test)
results = [int(round(x[0])) for x in predictions]  # round predictions

predictions = numpy.array(Y_test, dtype=int)
results = numpy.array(results, dtype=int)

print(predictions)
print(results)

correct = 0
incorrect = 0

for i in range(0, 18):
    if(predictions[i] == results[i]):
        correct = correct + 1
    else:
        incorrect = incorrect + 1

print("\nTESTING %s: %.2f%%" % ("acc", (correct/(correct + incorrect))*100))
