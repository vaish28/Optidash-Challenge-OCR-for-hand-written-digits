"""Importing libraries"""

import keras
from build_model import classifier_model
from training_data import load_data, preprocess_data

import matplotlib.pyplot as plt

"""Loading the dataset and preprocessing"""

X_train, y_train, X_test, y_test = load_data()
X_train_classifier, y_train_classifier, X_val_classifier, y_val_classifier, X_test_classifier, y_test_classifier = preprocess_data(X_train, y_train, X_test, y_test)


"""Model"""

classifier = classifier_model()
classifier.summary()

"""Specifying Callbacks and Training the Model"""

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>=0.98):
            print("Reached 98% accuracy so cancelling training!")
            self.model.stop_training = True

callback = myCallback()

with tf.device('/GPU:0'):
    history = classifier.fit(X_train_classifier, y_train_classifier, batch_size=32, epochs=100, verbose=1, validation_data = (X_val_classifier, y_val_classifier), callbacks = [callback])
"""Training Evaluation"""

# model training performance for accuracy
plt.plot(history.history['acc'][0:100])
plt.plot(history.history['val_acc'][0:100])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training_accuracy', 'validation_accuracy'], loc='lower right')
plt.show()

# model training performance for loss
plt.plot(history.history['loss'][0:100])
plt.plot(history.history['val_loss'][0:100])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training_loss', 'validation_loss'], loc='upper right')
plt.show()
