from keras.layers import Dense, Activation, MaxPooling2D, Flatten, Dropout, Conv2D
from keras.models import Sequential  
from keras.utils import np_utils
import pandas as pd 
import numpy as np

  
# reshape the data  
X_Train = trainX.reshape(len(trainX)/(20*36), 36, 20, 1).astype('float32')  
X_Test = testX.reshape(len(testX)/(20*36), 36, 20, 1).astype('float32')  
y_Train = trainY
y_Test = testY
  
model = Sequential()  
# Create Conv layer 1  
model.add(Conv2D(filters=1,  
                 kernel_size=(5,5),  
                 padding='same',  
                 input_shape=(36,20,1),  
                 activation='relu'))  

# Create Max-Pool 1  
model.add(MaxPooling2D(pool_size=(2,2)))  

# Add Dropout layer  
model.add(Dropout(0.25)) 

#flatten
model.add(Flatten())

#add fully connected
model.add(Dense(2, activation='softmax')) 

model.summary()  
print("")  



# 定義訓練方式  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
  
# 開始訓練  
train_history = model.fit(x=X_Train,  
                          y=y_Train, validation_split=0.2,  
                          epochs=5, batch_size=100, verbose=2)


#scores = model.evaluate(X_Test, y_Test)  
#print()  
#print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0)) 


predictY = model.predict_classes(X_Test)  # Making prediction and save result to prediction  
print()  

   