import numpy as np
import cv2
import crop
import lines

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # code, frame = crop.crop(frame)
    # if code != 0:
    #     print(code)
    #     continue
    code, frame, boxes = lines.find_lines(frame)
    
    if code != 0:
        continue
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()