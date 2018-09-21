# MACHINE LEARNING

# TODO

# probabilistic modelling (naive bayes, logistic regression)
# kernel methods (SVM - decision boundary - hyperplanes)
# random forrests
# gradient boosting machines
# regularization
# gradient descent
# cross entropy error
# backpropagation
# dropout
# roc auc

# 93

# ---------------------------------------------------------------------------------------------------------------------

# branches of machine learning
    # 1. supervised learning: known targets, annotations (sequence generation, object detection)
    # 2. unsupervised learning: no known targets (data analytics, clustering)
    # 3. self-supervised learning: annotattions aren't made by humans, but by the computer (heuristics)
    # 4. reinforcement learning: receives info about the environment, then chooses an action that maximizes the reward

# evaluation of machine-learning model = generalization
# parameters of neural networks = weights, hyperparameters of neural networks = number, size of layers
# 3 sets, train, validate, test -> to fine-tune the model for performance (hyperparameters), data from validation set

# validation types:
    # 1. hold-out validation: shuffle data, split labeled data into train and validation set
        # if the measurement varies because of shuffling, then you have too little data
    # 2. k-fold validation: split data into k sets, learn on k-1 subsets and validate on 1 subset, repeat for other
        # then average the scores, this solves to problem of too little data
    # 3. repeat k-fold validation multiple times, each time shuffle the data, this is time consuming but precise

# data preprocessing:
    # 1. data vectorization: transforms data into tensors (float, int, one-hot)
    # 2. normalization: float values (0,1), std = 0, mean = 1 (-1,1)
        # learning will be harder if values are big ints and features vary (one is in range (0,1) other (1,100)
        # x -= x.mean(axis=0)
        # x /= x.std(axis=0)
    # 3. missing data, if value 0 is not important in data add 0 instead of missing values
        # if training on data and test set has missing values, generate missing values also in training set

# feature engineering, making a problem easier to learn for the algorithm
# optimization refers to the process of adjusting a model to get the best performance possible on the training data
# generalization refers to how well the trained model performs on data it has never seen before

# regularization techniques:
    # 1. reduce network size (less layers, less neurons per layer -> less parameters, less memorization capacity)
        # therefore test different architectures on validation set, track loss, acc
    # 2. add weight regularization (weights have small values ->  distribution of weights is more regular)
        # model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))  # l2 = weight decay
    # 3. add dropout (applied to a layer, consists of randomly dropping out i.e. setting to zero
        # a number of output features of the layer during training, helps with prevention of random patterns)
        # dropout rate is the fraction of the features that are zeroed out (0.2 - 0.5)
        # model.add(layers.Dropout(0.5)), example: [0.2, 0.3, 0.9, 0.1] -> [0, 0.3, 0, 0.1]

# measure of success: for balanced-classification problems (ROC AUC), for inbalanced (precision, recall)
