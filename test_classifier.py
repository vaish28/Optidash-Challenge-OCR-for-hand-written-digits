
""" Importing libraries """
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report

""" Downloading the dataset and preprocessing it """

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_test_expanded =  np.expand_dims(X_test, axis=-1)
y_test_catagorical = keras.utils.to_categorical(y_test, 10)

X_test_classifier = X_test_expanded
y_test_classifier = y_test_catagorical

""" Loading the trained model """
model = keras.models.load_model("classifier.h5")
model.summary()

""" Confusion matrix and classification report """

y_true = np.argmax(y_test_classifier,axis=1)
y_p = model.predict(X_test_classifier)
y_predicted = np.argmax(y_p,axis=1)

print('confusion matrix')
print(confusion_matrix(y_true,y_predicted))

print('Classification report')
print(classification_report(y_true,y_predicted))

""" Visulal representation and confirmation """

test_image = X_test[101]

plt.imshow(test_image)
plt.show()

print("number_original: ", y_test[101])

model_image = np.expand_dims(test_image,-1)
model_image = np.expand_dims(model_image,0)

prediction = model.predict(model_image)
number_predicted = np.argmax(prediction,axis=1)

print("number_predicted: ", int(number_predicted))
