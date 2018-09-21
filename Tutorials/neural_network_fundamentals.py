# TUTORIAL
# http://www.cristiandima.com/neural-networks-from-scratch-in-python/

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

# #features = #input nodes
# #classes = #output nodes

# weight matrix = [n x d] = [#input nodes x #hidden nodes]
# input matrix = [s x n] = [#samples x #features]
# output matrix = [s x d] = input x weights

# AF softmax: normalizes each number of a vector to a value between 0 and 1
# they sum up to 1 (probability distribution) -> e^xi/sum across samples(e^x)

# error function: negative log likelihood
#  -1/s * sum across samples(sum across classes(real * log(predicted))
# log(x) if x is [0,1] is negative, reason for minus (negative log likelihood)
# real * log(predicted) -> if prediction is close to real, the result will be close to zero so almost no error
# if this is true across all samples (small average error), there will be no need to change weights

# gradient descent searches for minimum of function, takes small steps in the negative value of gradient
# gradient of loss function is dependant on partial derivatives of weights and biases

np.random.seed(1)

# generate three Gaussian clouds each holding 500 points
X1 = np.random.randn(500, 2) + np.array([0, -2])
X2 = np.random.randn(500, 2) + np.array([2, 2])
X3 = np.random.randn(500, 2) + np.array([-2, 2])

# put them all in a big matrix
X = np.vstack([X1, X2, X3])

# generate the one-hot-encodings
labels = np.array([0]*500 + [1]*500 + [2]*500)
T = np.zeros((1500, 3))
for i in range(1500):
    T[i, labels[i]] = 1

# visualize the data
plt.scatter(X[:,0], X[:,1], c=labels, s=100, alpha=0.5)
#plt.show()

samples = X.shape[0] # 1500 samples
features = X.shape[1] # 2 features
hidden_nodes = 5
classes = 3

# randomly initialize weights
W1 = np.random.randn(features, hidden_nodes)
b1 = np.random.randn(hidden_nodes)
W2 = np.random.randn(hidden_nodes, classes)
b2 = np.random.randn(classes)

alpha = 10e-6 # learning rate
costs = []

for epoch in range(10000):
    # forward pass
    A = sigmoid(X.dot(W1) + b1) # A = sigma(Z)
    Y = softmax(A.dot(W2) + b2) # Y = softmax(Z2)

    # backward pass
    delta2 = Y - T
    delta1 = (delta2).dot(W2.T) * A * (1 - A)

    W2 -= alpha * A.T.dot(delta2)
    b2 -= alpha * (delta2).sum(axis=0)

    W1 -= alpha * X.T.dot(delta1)
    b1 -= alpha * (delta1).sum(axis=0)

    if epoch % 1000 == 0:
        loss = np.sum(-T * np.log(Y))
        print('Loss function value: ', loss)
        costs.append(loss)