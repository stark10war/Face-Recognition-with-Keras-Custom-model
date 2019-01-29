# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:13:27 2019

@author: Shashank
"""

import os

os.chdir('this folder path')
import numpy as np
import pandas as pd
import cv2

## Setting the base directory
BASE_DIR = os.path.dirname(os.path.abspath('face-train.py'))
image_dir =  os.path.join(BASE_DIR, "Input_images")
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')

person_name = input("Please Enter your name: ") ##Get input label from the user
os.mkdir(os.path.join(image_dir,person_name)) #Create directory for the person to save images
image_count = 0 #initiated image_count
cap = cv2.VideoCapture(0) # Video capture instance
while True:
    check,frame = cap.read() 
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convert to gray
    faces_coords = face_cascade.detectMultiScale(frame_gray, scaleFactor = 1.05, minNeighbors = 5) # Detect face Cordinates
    for x,y,w,h in faces_coords:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) # Draw rectangle
        face_gray = frame_gray[y:y+h, x:x+w]
        face_resized =  cv2.resize(face_gray, (50,50))
        out_path = os.path.join(image_dir,person_name,person_name+str(image_count)+'.jpg')
        key = cv2.waitKey(1)
        if key == ord('c'):
            image_count = image_count+1
            cv2.imwrite(out_path,face_resized)
            print('face' + str(image_count)+ ' saved on disk' )
        
   
    cv2.imshow('frame',frame)
    if key == ord('q') or image_count == 20 :
        break

cap.release()
cv2.destroyAllWindows()





   
