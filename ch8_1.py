#!/usr/bin/env python3
from keras.datasets import mnist
from keras.utils import np_utils
from utils import *
import numpy as np
import pandas as pd
from keras.models import model_from_json
np.random.seed(10)

# Constants
KERAS_MODEL_NAME = 'mnist_model_cnn.model'
KERAS_MODEL_WEIG = 'mnist_model_cnn.h5'

# Read MNIST data
(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()

# Translation of feature data
X_Train4D = X_Train.reshape(X_Train.shape[0], 1, 784, 1).astype('float32')
X_Test4D = X_Test.reshape(X_Test.shape[0], 1, 28, 1).astype('float32')


# Standardize feature data
X_Train4D_norm = X_Train4D / 255
X_Test4D_norm = X_Test4D /255

# Label Onehot-encoding
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()
# Create CN layer 1
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))
# Create Max-Pool 1
model.add(MaxPooling2D(pool_size=(2,2)))

# Create CN layer 2
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))

# Create Max-Pool 2
model.add(MaxPooling2D(pool_size=(2,2)))

# Add Dropout layer
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()
print("")

use_exist_model = False
if os.path.isfile(KERAS_MODEL_NAME):
    train_history = None
    with open(KERAS_MODEL_NAME, 'r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(KERAS_MODEL_WEIG)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    use_exist_model = True
else:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(x=X_Train4D_norm,
                              y=y_TrainOneHot, validation_split=0.2,
                              epochs=10, batch_size=300, verbose=2)

if isDisplayAvl() and train_history:
    show_train_history(train_history, 'acc', 'val_acc')
    show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(X_Test4D_norm, y_TestOneHot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

print("\t[Info] Making prediction of X_Test4D_norm")
prediction = model.predict_classes(X_Test4D_norm)  # Making prediction and save result to prediction
print()
print("\t[Info] Show 10 prediction result (From 240):")
print("%s\n" % (prediction[240:250]))

if isDisplayAvl():
    plot_images_labels_predict(X_Test, y_Test, prediction, idx=240)


print("\t[Info] Display Confusion Matrix:")
print("%s\n" % pd.crosstab(y_Test, prediction, rownames=['label'], colnames=['predict']))

# Serialized model
if not use_exist_model:
    print("\t[Info] Serialized Keras model to %s..." % (KERAS_MODEL_NAME))
    with open(KERAS_MODEL_NAME, 'w') as f:
        f.write(model.to_json())
    model.save_weights(KERAS_MODEL_WEIG)
print("\t[Info] Done!")
