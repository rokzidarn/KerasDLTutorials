# TUTORIAL
# https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

def plot_decision_boundary(pred_func):  # plot boundary after classifying
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def logistic_regression_classifier(X,y):
    clf = sklearn.linear_model.LogisticRegressionCV()  # logistic regression, linear classifier
    # needs non-linear features to be more accurate, as a polynomial
    clf.fit(X, y)

    plot_decision_boundary(lambda x: clf.predict(x))
    plt.title("Logistic Regression")
    plt.show()

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    # init params to random
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    # learning, gradient descent to minimize error
    for i in range(0, num_passes):

        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax
        # probabilites for each class, sum = 1, [prob_c1, prob_c2]

        # backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # optional; add regularization terms to loss function
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # optional; gradient descent parameter update, to decrease loss function
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

    # assign new parameters to the model
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return model

# MAIN

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)  # set plot parameters

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)  # generate data
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)  # plot data

# logistic_regression_classifier(X,y)

# NEURAL NETWORK

# number of input nodes are defined by the dimensionality of data; (x,y) coordinates
# number of output nodes are defined by number of prediction classes; male/female

# number of hidden layers defines how complex the learning function are, but it can cause overfitting
# the activation function converts input into output (tanh, sigmoid, ReLU)

# predictions:
    # z1 = x*w1+b1 (weighted sum of input, b is bias)
    # a1 = tanh(z1)
    # z2 = a1*w2+b2
    # prediction = a2 = softmax(z2)

# to calculate the loss we use cross-entropy loss
# this value we must minimize so we use gradient descent
# to calculate the gradient descent we need all of the gradients of paramaters (w1, w2, b1, b2), this is backpropagation

num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

# gradient descent parameters
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Build a model with a 3 hidden layers
model = build_model(3, print_loss=True)

plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()