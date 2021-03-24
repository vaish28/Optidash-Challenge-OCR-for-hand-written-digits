""" python file containing function related to MNIST dataset download and preprocessing"""

import keras
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):

    y_train_catagorical = keras.utils.to_categorical(y_train, 10)
    y_test_catagorical = keras.utils.to_categorical(y_test, 10)

    X_train_expanded =  np.expand_dims(X_train, axis=-1)
    X_test_expanded =  np.expand_dims(X_test, axis=-1)

    X_train_classifier, X_val_classifier, y_train_classifier, y_val_classifier = train_test_split(X_train_expanded, y_train_catagorical, test_size = 0.1, random_state = 1)

    X_test_classifier = X_test_expanded
    y_test_classifier = y_test_catagorical

    return X_train_classifier, y_train_classifier, X_val_classifier, y_val_classifier, X_test_classifier, y_test_classifier
