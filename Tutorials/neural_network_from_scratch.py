import numpy as np

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

# MAIN
# https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/

X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])  # 3 examples with 4 features
y = np.array([[1], [1], [0]])  # real result of each example
output = []

# initialization
epoch = 5000
lr = 0.1  # learning rate

input_layer_neurons = X.shape[1]  # number of features in data set
hidden_layer_neurons = 3  # number of hidden layers neurons
output_neurons = 1  # number of neurons at output layer

# weight and bias initialization
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))

w_out = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
b_out = np.random.uniform(size=(1, output_neurons))

for i in range(epoch):
    # forward
    hidden_layer_input = np.dot(X, wh) + bh  # from input to hidden layer
    hidden_layer_activations = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_activations, w_out) + b_out
    output = sigmoid(output_layer_input)  # from hidden layer to output

    # backpropagation:
    #  1. error = real - prediction
    #  2. delta = error * AF of prediction derivative
    # optional: if multiple layers, propagate error = dot(output weights, delta)
    # then repeat step 2 with propagated error
    #  3. new weight = old weight + dot(input, delta) * learning rate (alpha)

    Error = y-output  # compare prediction to real value

    # gradients, check how close we are to solution, the closer we are to 0 or 1 the less will the gradient be
    # meaning less correction on weights
    slope_output_layer = derivatives_sigmoid(output)  # minimize error with gradient
    slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)

    delta_output = Error * slope_output_layer  # delta (change) factor of output, how to modify weights of w_out
    Error_at_hidden_layer = delta_output.dot(w_out.T)  # propagate error to hidden layer
    delta_hidden_layer = Error_at_hidden_layer * slope_hidden_layer  # delta (change) factor of hidden layer

    w_out = w_out + hidden_layer_activations.T.dot(delta_output) * lr  # fix weights and bias
    b_out = b_out + np.sum(delta_output, axis=0, keepdims=True) * lr

    wh = wh + X.T.dot(delta_hidden_layer) * lr
    bh = bh + np.sum(delta_hidden_layer, axis=0, keepdims=True) * lr

print(output)