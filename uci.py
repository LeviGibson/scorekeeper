import cv2
import numpy as np
import crop
import lines

cap = cv2.VideoCapture(0)


while True:
    input("Press enter to scan")
    
    ret, frame = cap.read()
    
    frame = lines.find_lines(frame)
    # code, frame = crop.crop(frame)
    # if code != 0:
    #     print("Error ({}) while scanning".format(code))
    #     continue
    
    crop.display(frame)
    