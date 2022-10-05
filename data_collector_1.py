import cv2
import os
import time
import uuid
import datetime
import argparse

parser = argparse.ArgumentParser(description='')
print("initializing camera")
time.sleep(3)

cap = cv2.VideoCapture('/dev/video0')



print()
cap.set(3,800)
cap.set(4,600)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

count = 0
buf=0
os.system("v4l2-ctl -d /dev/video0 -c  focus_absolute=0")
while(1):
    
    ret, image = cap.read()
    if cap.get(28)!=0:
        os.system("v4l2-ctl -d /dev/video0 -c focus_absolute=0")   
        print("focus is gone")
   
    if(buf % 5 == 0): 
        res = cv2.imwrite("./collect/cam1/cam_1_frame%d.jpg" % count, image)
        print(res)
        
       
        print("focus check cv2.CAP_PROP_FOCUS,: '{}'".format(cap.get(28)))
        print('Saved frame number :', count)
        count += 1
    buf+=1   

print("focus check cv2.CAP_PROP_FOCUS,: '{}'".format(cap.get(28)))
cap.release() 
cv2.destroyAllWindows()
