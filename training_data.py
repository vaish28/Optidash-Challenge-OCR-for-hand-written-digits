""" python file containing function related to MNIST dataset download and preprocessing"""

import keras
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):

    #converting integers to float values
    X_train_normal = X_train.astype('float32')
    X_test_normal = X_test.astype('float32')

    # normalizing the range between 0-1
    X_train_normal = X_train_normal/255.0
    X_test_normal = X_test_normal/255.0

    # changing the labels to categorical format for providing labels to the model for training
    y_train_catagorical = keras.utils.to_categorical(y_train, 10)
    y_test_catagorical = keras.utils.to_categorical(y_test, 10)

    # correcting the imagedataset dimenssion for model
    X_train_expanded =  np.expand_dims(X_train_normal, axis=-1)
    X_test_expanded =  np.expand_dims(X_test_normal, axis=-1)

    # splitting the X_training dataset in train and validation dataset for training purpose
    X_train_classifier, X_val_classifier, y_train_classifier, y_val_classifier = train_test_split(X_train_expanded, y_train_catagorical, test_size = 0.1, random_state = 1)

    X_test_classifier = X_test_expanded
    y_test_classifier = y_test_catagorical

    return X_train_classifier, y_train_classifier, X_val_classifier, y_val_classifier, X_test_classifier, y_test_classifier
