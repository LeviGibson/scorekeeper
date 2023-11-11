import numpy as np
import cv2
import crop
import lines
import reader

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
    
    times = lines.crop_boxes(frame, boxes)
    # cv2.imshow('frame', frame)
    
    # for id, i in enumerate(times):
    #     cv2.imwrite(str(id) + ".png", i)
    
    reader.read_time(times[2])
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()