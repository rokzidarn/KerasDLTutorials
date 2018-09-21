import pandas
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

x = np.array(12)  # scalar
print(x)
print(x.ndim)

x = np.array([12, 3, 6, 0])  # vector
print(x)
print(x.ndim)

x = np.array([[12, 3, 6, 0],[1, 37, 66, 10]])  # matrix
print(x)
print(x.ndim)
print(x.reshape(8,1))

x = np.array([[[12, 3, 6, 0],[1, 37, 66, 10]],[[1, 31, 6, 40],[11, 7, 26, 12]]])  # 3D tensor
print(x)
print(x.ndim)
print(x.shape)

# ndim = number of axes
# shape = number of dimensions per axis
# dtype = data type; uint8, float32, float64

#print(np.array([1,0,0,1]))
#print(np.array([[1,0,0,1],[1,1,1,1]]))

#print(np.sum([1,2]))
#print(np.sum([[1,2],[1,3]]))

#print(np.exp([1,-1]))
#print(np.exp([[1,-1],[1,2]]))

#print(np.tanh([1,-1]))
#print(np.tanh([[1,-1],[1,2]]))

#print(np.argmax([1,3]))

#print(np.zeros(2))
#print(np.zeros([3,3]))

#print(np.random.random(2))
#print(np.random.random([2,3]))

#print(np.dot(3,4))
#print(np.dot([1,-1],[1,2]))
#print(np.dot([[1,-1],[1,2]], [[1,-1],[1,2]]))

#print(np.transpose([[1,2,3],[4,5,6]]))

# load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# summarize dataset
print(dataset.shape)  # dimensions
print(dataset.head(10))  # first 10 lines
print(dataset.describe())  # summary of each attribute, min, max, avg
print(dataset.groupby('class').size())  # class distribution

# visualization
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
dataset.hist()
plt.show()

Dense(512, activation='relu')  # tensor -> relu(dot(weights, input) + b) -> tensonr

# KERAS NN LIFECYCLE

X = []  # data
y = []
z = []

model = Sequential()  # 1 input layer (2 inputs) + 1 hidden layer (5 neurons) + 1 output layer

model.add(Dense(5, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))  # binary, multiclass classification use softmax

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])  # stochastic gradient descent + mean square error
# binary classification binary CE, multiclass classification use categorical CE

history = model.fit(X, y, batch_size=10, epochs=100)
# each epoch can be partitioned into groups of input-output pattern pairs called batches

loss, accuracy = model.evaluate(X, y)

predictions = model.predict(z)