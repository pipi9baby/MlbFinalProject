from keras.layers import Dense, Activation, MaxPooling2D, Flatten, Dropout, Conv2D, Conv1D, AveragePooling1D, AveragePooling2D
from keras.models import Sequential  
from keras.utils import np_utils
from keras import backend as K
from keras import initializers
from keras.initializers import VarianceScaling
import keras
import pandas as pd
import numpy as np
import math
import json

init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)

path = r"D:\downloads\structure96"

fr = open("%s/trainX.json" %(path), "r")
trainX = np.array(json.loads(fr.read()))
fr = open("%s/trainY.json" %(path), "r")
trainY = np.array(json.loads(fr.read()))
fr = open("%s/SPEuk.json" %(path), "r")
SPEuk = np.array(json.loads(fr.read()))
fr = open("%s/TMEuk.json" %(path), "r")
TMEuk = np.array(json.loads(fr.read()))
fr = open("%s/NCEuk.json" %(path), "r")
NCEuk = np.array(json.loads(fr.read()))

#pair data
'''fr = open("%s/trainX_pair.json" %(path), "r")
trainX_pair = np.array(json.loads(fr.read()))
fr = open("%s/SPEuk_pair.json" %(path), "r")
SPEuk_pair = np.array(json.loads(fr.read()))
fr = open("%s/TMEuk_pair.json" %(path), "r")
TMEuk_pair = np.array(json.loads(fr.read()))
fr = open("%s/NCEuk_pair.json" %(path), "r")
NCEuk_pair = np.array(json.loads(fr.read()))
trainX = trainX + trainX_pair
SPEuk = SPEuk + SPEuk_pair
TMEuk = TMEuk + TMEuk_pair
NCEuk = NCEuk + NCEuk_pair'''


K.clear_session()
np.random.seed(10) 

# reshape the data
X_Train = trainX.reshape(len(trainX)//(20 * 96), 96, 20, 1).astype('float32')
SPEuk_Test = SPEuk.reshape(len(SPEuk)//(20 * 96), 96, 20, 1).astype('float32')
TMEuk_Test = TMEuk.reshape(len(TMEuk)//(20 * 96), 96, 20, 1).astype('float32')
NCEuk_Test = NCEuk.reshape(len(NCEuk)//(20 * 96), 96, 20, 1).astype('float32')
y_Train = trainY
  
model = Sequential()

# Create Conv layer 1  
model.add(Conv2D(filters=100, kernel_size=(8,5), padding='same', input_shape=(96, 20, 1), activation='relu', bias_initializer=init, data_format='channels_last'))  
#model.add(Conv2D(filters=40, kernel_size=(5, 5), padding='same', input_shape=(36, 20, 1), activation='relu', bias_initializer=init))  
# Create Max-Pool 1  
model.add(AveragePooling2D(pool_size=(2,2), strides=2))  

# Create Conv layer 2  
model.add(Conv2D(filters=80, kernel_size=(8,5), padding='same', activation='relu', bias_initializer=init))  
#model.add(Conv2D(filters=20, kernel_size=(5, 5), padding='same', activation='relu', bias_initializer=init))  
# Create Max-Pool 2  
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

# Create Conv layer 3 
model.add(Conv2D(filters=80, kernel_size=(8,5), padding='same', activation='relu', bias_initializer=init))
#model.add(Conv2D(filters=10, kernel_size=(5, 5), padding='same', activation='relu', bias_initializer=init)) 
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))

# Create Max-Pool 3
model.add(AveragePooling2D(pool_size=(2,2), strides=2)) 


# Add Dropout layer  
model.add(Dropout(0.25)) 

# flatten
model.add(Flatten())

# hidden layer
#model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu', bias_initializer=init))

# Add Dropout layer  
model.add(Dropout(0.5)) 

# add fully connected
model.add(Dense(2, activation='softmax', bias_initializer=init)) 



model.summary()

# difine method for training.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
  
# train
model.fit(x=X_Train, y=y_Train, validation_split=0.2, epochs=10, batch_size=60, verbose=2)

#save model
model.save('model_12.h5')


TP = 0; FN = 0; TN = 0; FP = 0;
SPEuk_predict = model.predict_classes(SPEuk_Test)
for i in SPEuk_predict:
    if i == 1: TP += 1
    else: FP += 1
TMEuk_predict = model.predict_classes(TMEuk_Test)
for i in TMEuk_predict: 
    if i == 0: TN += 1
    else: FN += 1
NCEuk_predict = model.predict_classes(NCEuk_Test)
for i in NCEuk_predict:
    if i == 0: TN += 1
    else: FN += 1

print(TP, TN, FN, FP)
MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print(MCC)
