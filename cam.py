import cv2
import os
import random

from mamt import waste_prediction

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 640) # set video height
while(True):
    ret, img = cam.read()
    cv2.imshow('image', img)
    

    k = cv2.waitKey(33) & 0xff # Press 'ESC' for exiting video
    if k == 32:
        print('CALLING')
        # Save Image to TEMP
        id = random.randrange(0, 999999999, 2)
        fn = f'TEMP/{id}.jpg'
        cv2.imwrite(fn, img)
        res = waste_prediction(fn)
        print(res)
    if k == 27:
        break