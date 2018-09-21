import numpy as np

# 2 Layer NN - 1 input layer + 1 hidden layer
# https://iamtrask.github.io/2015/07/12/basic-python-network/

# sigmoid function, activation function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# we use the derivative of the sigmoid in backpropagation,
# but since it is already calculated before (forward pass -> weight * input)
# we can use the value for faster computation of the derivative

# MAIN

# DATA
# input dataset, each row is a training example (4 examples, 3 input nodes)
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset, result of the each training example (4 examples, 4 rows after transpose)
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic, randomly distributed in exactly the same way each time
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1  # layer of weights, connecting l0 and l1
# range of random values (-1, 1), 3x1 dimension (3 inputs 1 output)
# (n1) ->
# (n2) -> -> (nr)
# (n3) ->

for iter in range(10000):  # repetitions of training code to optimize weights for predictions

    # forward propagation
    l0 = X  # first layer of NN, specified by input data, all examples at the same time (full batch)
    l1 = nonlin(np.dot(l0, syn0))  # second layer of NN, hidden layer (example * weight)
    # matrix multiplication
    # e11 e12 e13               f1              r1
    # e21 e22 e23       DOT     f2      =       r2
    # e31 e32 e33               f3              r3
    # e41 e42 e43                               r4

    # row * column
    # example * weight = prediction of the example
    # e11*f1 + e12*f2 + e13*f3 = r1

    # sigmoid defines the prediction

    # error, how much did we miss
    l1_error = y - l1

    # multiply how much we missed by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)  # reduce the error of the prediction

    # update weights, we update each weight by the input and by how much we missed the output
    # greater the miss, larger error, the larger the adjustment of the weight
    syn0 = syn0 + np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)