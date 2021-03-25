""" python file containing function related to CNN+MLP model construction, specifications and compilation """

import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

def classifier_model():

    # initializing the sequential model
    model = Sequential()

    # adding first bolck of the model
    model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu', kernel_initializer='random_normal'))
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'random_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    # adding second block of the model
    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'random_normal'))
    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'random_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    # Flattening the model
    model.add(Flatten())

    # adding the Dense layers
    model.add(Dense(units = 128, activation = 'relu', kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # adding the output layer
    model.add(Dense(units = 10, activation = 'softmax', kernel_initializer='random_normal'))

    # model compilation + optimizer and loss function specifications.
    opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)

    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

    return model
