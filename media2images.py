import numpy as np
import cv2

#filepath = '~/Downloads/2person_35.h264'
filepath = './trt_pose/tasks/human_pose/video_sample_34_1.mp4'
cap = cv2.VideoCapture(filepath)

while(cap.isOpened()):
    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow('frame',rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
