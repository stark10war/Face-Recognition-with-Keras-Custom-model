# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:15:53 2019

@author: Shashank
"""

import os
os.chdir('D:\\Python practice\\keras practice')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import _pickle as pickel
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing import image

PATH = os.getcwd()
image_dir =  os.path.join(PATH, "Input_images")

y_labels = []
face_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root,file)
        label = os.path.basename(root)
        print(label,path)
        img = image.load_img(path, target_size=(50,50), color_mode='grayscale')        
        img = image.img_to_array(img)
        y_labels.append(label)
        face_train.append(img)


train = np.array(face_train)

train = train/255



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y_labels = labelencoder.fit_transform(y_labels)
pickel.dump(labelencoder,  open('labelencoder.dat', 'wb'))


labels_onehot = keras.utils.to_categorical(y_labels, num_classes=5)

from sklearn.model_selection import train_test_split 

xtrain, xtest, ytrain, ytest = train_test_split(train,labels_onehot, test_size = 0.3, random_state = 24)

im_shape = (50,50,1)




## Keras model Structure
import random
random.seed(24)


Model = Sequential()

Model.add(Conv2D(40,kernel_size=(3,3),activation= 'relu', input_shape= im_shape))
Model.add(Conv2D(80,kernel_size=(3,3),activation= 'relu'))
Model.add(MaxPooling2D(pool_size = (2,2)))
Model.add(Conv2D(140,kernel_size=(3,3),activation= 'relu'))
Model.add(MaxPooling2D(pool_size = (2,2)))

Model.add(Flatten())
Model.add(Dense(250, activation='relu'))
Model.add(Dense(128, activation='relu'))
Model.add(Dropout(0.25))
Model.add(Dense(5, activation='softmax'))
Model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

Model.summary()

Model.fit(xtrain,ytrain, batch_size= 70, epochs=30,verbose=1, validation_data=(xtest,ytest), shuffle= True)

Model.save('Keras_Model_face.h5')


#scores = Model.evaluate(xtest,ytest)


#predicted = Model.predict_classes(xtest)
#actual = np.argmax(ytest, axis=1)

#from sklearn.metrics import accuracy_score, confusion_matrix
#confusion_matrix(actual,predicted)




