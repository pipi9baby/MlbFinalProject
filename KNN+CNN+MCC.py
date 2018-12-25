from keras.layers import Dense, Activation, MaxPooling2D, Flatten, Dropout, Conv2D
from keras.models import Sequential  
from keras.utils import np_utils
from keras import backend as K
import pandas as pd
import numpy as np
import math
import json

path = "/home/mlb/users/q56074077/course/signalp/KNNData"

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

K.clear_session()
np.random.seed(10) 

# reshape the data
X_Train = trainX.reshape(len(trainX)//(20 * 36), 36, 20, 1).astype('float32')
SPEuk_Test = SPEuk.reshape(len(SPEuk)//(20 * 36), 36, 20, 1).astype('float32')
TMEuk_Test = TMEuk.reshape(len(TMEuk)//(20 * 36), 36, 20, 1).astype('float32')
NCEuk_Test = NCEuk.reshape(len(NCEuk)//(20 * 36), 36, 20, 1).astype('float32')
y_Train = trainY
  
model = Sequential()
# Create Conv layer 1  
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(36, 20, 1), activation='relu'))  

# Create Max-Pool 1  
model.add(MaxPooling2D(pool_size=(2, 2)))  

# Create Conv layer 2  
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))  

# Create Max-Pool 2  
model.add(MaxPooling2D(pool_size=(2, 2)))

# Create Conv layer 3 
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))  

# Create Max-Pool 3
model.add(MaxPooling2D(pool_size=(2, 2))) 

# Add Dropout layer  
model.add(Dropout(0.25)) 

# flatten
model.add(Flatten())

# hidden layer
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Add Dropout layer  
model.add(Dropout(0.5)) 

# add fully connected
model.add(Dense(2, activation='softmax')) 

model.summary()

# difine method for training.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
  
# train
model.fit(x=X_Train, y=y_Train, validation_split=0.2, epochs=5, batch_size=100, verbose=2)

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
