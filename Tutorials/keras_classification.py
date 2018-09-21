# KERAS TUTORIAL
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# MAIN

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("Data/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables -> ONE HOT VECTOR
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)  # debug off, no epoch print

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)  # shuffle data, then partition on 1/10

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))