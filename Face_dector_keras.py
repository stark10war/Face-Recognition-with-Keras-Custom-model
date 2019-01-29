# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:01:35 2019

@author: Shashank
"""

import os
os.chdir('D:\\Python practice\\keras practice')
import numpy as np
import cv2
import _pickle as pickel
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing import image


face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')

recogniser = keras.models.load_model('Keras_Model_face.h5')
labelencoder = pickel.load( open('labelencoder.dat', 'rb'))

cap = cv2.VideoCapture(0)
while True:
    check,frame = cap.read() 
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces_coords = face_cascade.detectMultiScale(frame_gray, scaleFactor = 1.05, minNeighbors = 5)
    for x,y,w,h in faces_coords:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        face = frame_gray[y:y+h, x:x+w]
        face_resized =  cv2.resize(face, (50,50))
        face_final = np.reshape(face_resized,(1,50,50,1))
        pred = recogniser.predict_classes(face_final)
        prediction = labelencoder.inverse_transform(pred)
        cv2.putText(frame,prediction[0] ,(int(x+w/2),y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255), 1)
        
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()




